#!/usr/bin/env python3
"""Profile one current cached-decode workload with a process phase delta."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--many-step-count", type=int, default=3)
    parser.add_argument("--rocprof", default="rocprofv3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.batch <= 0 or args.decode_tokens <= 0 or
            args.warmup < 0 or args.many_step_count <= 1):
        parser.error("cached decode profile inputs are invalid")
    return args


def model(path: Path, name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("models", []) if document.get("schema_version") == 1 else []
    selected = [row for row in rows if row.get("name") == name]
    if len(selected) != 1:
        raise RuntimeError("profile needs exactly one pinned selected model")
    required = {"revision", "config", "weights", "parameter_count", "inference"}
    if required - selected[0].keys():
        raise RuntimeError("profile model contract is incomplete")
    return selected[0]


def repeated(seed: list[int], length: int) -> str:
    if not seed or any(int(value) < 0 for value in seed):
        raise RuntimeError("profile token seed is invalid")
    return ",".join(str(seed[index % len(seed)]) for index in range(length))


def app_command(args: argparse.Namespace, selected: dict, steps: int) -> list[str]:
    return [
        str(args.binary), "--config", selected["config"],
        "--weights", selected["weights"], "--tokens",
        repeated(selected["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(args.batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", str(args.context + args.decode_tokens),
        "--new-tokens", str(args.decode_tokens),
        "--warmup", str(args.warmup), "--steps", str(steps),
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(steps),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--workload", "decode",
    ]


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("profiled application emitted no JSON")


def require_stats(directory: Path, kind: str) -> Path:
    matches = list(directory.glob(f"*_{kind}_stats.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"rocprof emitted {len(matches)} {kind} stats files")
    return matches[0]


def main() -> int:
    args = options()
    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise RuntimeError("output directory is not empty; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = model(args.manifest, args.model)
    records = []
    kernel_stats: dict[int, Path] = {}
    with tempfile.TemporaryDirectory(
            prefix="microllm-current-cached-profile-") as temporary:
        temporary_root = Path(temporary)
        for steps in (1, args.many_step_count):
            trace = temporary_root / f"{steps}-step"
            trace.mkdir()
            command = [
                args.rocprof,
                "--kernel-trace", "--hip-runtime-trace",
                "--memory-copy-trace", "--memory-allocation-trace",
                "--stats", "--output-format", "csv",
                "--output-file", "trace", "--output-directory", str(trace),
                "--", *app_command(args, selected, steps),
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            record = last_json(completed.stdout)
            expected = {
                "status": "pass", "model": args.model,
                "context": args.context, "batch": args.batch,
                "decode_tokens": args.decode_tokens,
                "use_cache": True, "kv_cache_dtype": "bf16",
                "decode_mode": "steady", "workload": "decode",
                "warmup": args.warmup, "steps": steps,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError("profiled cached decode contract changed")
            if (record.get("decode_step_semantics") !=
                    "one_model_forward_per_measured_token" or
                    record.get("requested_cache_capacity") !=
                    args.context + args.decode_tokens or
                    record.get("kv_cache_utilization") != 1.0 or
                    record.get("measured_forward_steps") !=
                    args.batch * args.decode_tokens * steps):
                raise RuntimeError("profiled decode accounting changed")
            record.update({
                "record_type": "current_cached_decode_profile_run",
                "model_revision": selected["revision"],
                "profile_steps": steps,
            })
            records.append(record)
            (output / f"{steps}-step-run.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            for kind in ("kernel", "hip_api", "memory_copy",
                         "memory_allocation"):
                source = require_stats(trace, kind)
                destination = output / f"{steps}-step-{kind.replace('_', '-')}-stats.csv"
                shutil.copy2(source, destination)
                if kind == "kernel":
                    kernel_stats[steps] = destination

    delta = subprocess.run([
        sys.executable, str(Path(__file__).with_name("profile_step_delta.py")),
        "--one-step", str(kernel_stats[1]),
        "--many-step", str(kernel_stats[args.many_step_count]),
        "--many-step-count", str(args.many_step_count),
        "--output-directory", str(output),
        "--track", "inference_cached_decode_kernel_phase_delta",
    ], text=True, capture_output=True, check=False)
    if delta.returncode != 0:
        raise RuntimeError(delta.stdout + delta.stderr)
    profile = json.loads((output / "profile-delta.json").read_text(
        encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "current_cached_decode_profile_summary",
        "model": args.model,
        "model_revision": selected["revision"],
        "context": args.context,
        "batch": args.batch,
        "decode_tokens": args.decode_tokens,
        "warmup": args.warmup,
        "one_step_count": 1,
        "many_step_count": args.many_step_count,
        "derived_generations": args.many_step_count - 1,
        "derived_forward_steps": args.batch * args.decode_tokens,
        "kernel_profile": profile,
        "application_runs": records,
        "decision": "select the largest current cached-decode kernel category",
    }
    (output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"profile_current_cached_decode: {error}", file=sys.stderr)
        raise SystemExit(2)
