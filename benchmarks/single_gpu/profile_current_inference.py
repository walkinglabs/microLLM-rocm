#!/usr/bin/env python3
"""Profile the retained B1T1024 inference path with load-subtracted traces."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


INDICES = {
    "qwen2.5-0.5b": (64755, 65200, 24),
    "deepseek-r1-distill-qwen-1.5b": (64755, 65212, 28),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--rocprof", default="rocprofv3")
    parser.add_argument("--expected-bf16-ffn-norm", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file() or not args.binary.is_file():
        parser.error("profile inputs are unavailable")
    return args


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != set(INDICES):
        raise RuntimeError("current inference profile requires pinned models")
    return result


def repeated(seed: list[int], length: int) -> str:
    return ",".join(str(seed[index % len(seed)]) for index in range(length))


def app_command(args: argparse.Namespace, model: dict, steps: int,
                logits: Path) -> list[str]:
    qkv_index, gate_up_index, _ = INDICES[model["name"]]
    return [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        repeated(model["inference"]["token_ids"], 1024),
        "--device", "hip", "--batch", "1", "--top-k", "10",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true", "--bf16-ffn-arena-minimum-rows", "1024",
        "--bf16-qkv-arena", "true", "--bf16-qkv-arena-minimum-rows", "1024",
        "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
        "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
        "--inference-bthd-attention", "true",
        "--inference-bthd-bf16-qk", "true",
        "--inference-bthd-online-attention", "false",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", str(steps),
        "--prefill-logits", "last", "--logits-output", str(logits),
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


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    summaries = []
    with tempfile.TemporaryDirectory(prefix="microllm-current-profile-") as temporary:
        root = Path(temporary)
        for model in models(args.manifest):
            model_output = args.output_directory / model["name"]
            model_output.mkdir(parents=True, exist_ok=True)
            stats: dict[int, Path] = {}
            for steps in (1, 6):
                trace = root / f"{model['name']}-{steps}"
                trace.mkdir()
                logits = trace / "logits.bin"
                command = [
                    args.rocprof, "--kernel-trace", "--stats",
                    "--output-format", "csv", "--output-file", "trace",
                    "--output-directory", str(trace), "--",
                    *app_command(args, model, steps, logits),
                ]
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stdout + completed.stderr)
                record = last_json(completed.stdout)
                if record.get("status") != "pass" or \
                        record.get("inference_bthd_attention") is not True or \
                        record.get("inference_bthd_bf16_qk") is not True or \
                        record.get("inference_bthd_online_attention") is not False:
                    raise RuntimeError("profiled default route changed")
                if args.expected_bf16_ffn_norm and \
                        record.get("bf16_ffn_norm_fusion_enabled") is not True:
                    raise RuntimeError("profiled FFN Norm default route changed")
                record.update({
                    "record_type": "current_inference_profile_run",
                    "model": model["name"], "revision": model["revision"],
                    "prefill_steps_profiled": steps,
                })
                records.append(record)
                (model_output / f"{steps}-step-run.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
                source = trace / "trace_kernel_stats.csv"
                if not source.is_file():
                    raise RuntimeError("rocprof did not emit kernel stats")
                destination = model_output / f"{steps}-step-kernel-stats.csv"
                shutil.copy2(source, destination)
                stats[steps] = destination
            delta = subprocess.run([
                sys.executable,
                str(Path(__file__).with_name("profile_step_delta.py")),
                "--one-step", str(stats[1]), "--many-step", str(stats[6]),
                "--many-step-count", "6", "--output-directory", str(model_output),
                "--track", "inference_prefill_kernel_phase_delta",
            ], text=True, capture_output=True, check=False)
            if delta.returncode != 0:
                raise RuntimeError(delta.stdout + delta.stderr)
            profile = json.loads((model_output / "profile-delta.json").read_text())
            profile.update({"model": model["name"], "revision": model["revision"]})
            (model_output / "profile-delta.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            summaries.append(profile)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": ("post_bf16_ffn_norm_profile_summary"
                        if args.expected_bf16_ffn_norm else
                        "current_inference_profile_summary"),
        "bf16_ffn_norm_fusion_expected": args.expected_bf16_ffn_norm,
        "profile_processes": len(records),
        "sequence": 1024,
        "batch": 1,
        "derived_forwards": 5,
        "models": summaries,
        "decision": ("reselect hotspot after retained FFN Norm fusion"
                     if args.expected_bf16_ffn_norm else
                     "keep softmax local track closed; screen exact T1024 Attention GEMM solutions"),
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
