#!/usr/bin/env python3
"""Gate DeepSeek rows2 grouped gate/up on current cached decode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "grouped_decode_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMMON)

POLICIES = ("arena", "grouped65193")
SOLUTION = 65193


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not args.manifest.is_file() or not args.binary.is_file() or \
            args.runs != 3 or args.timeout_seconds <= 0:
        parser.error("grouped decode model inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args, model: dict, policy: str, warmup: int, steps: int,
            logits: Path | None = None) -> list[str]:
    ids = COMMON.expanded(model["inference"]["token_ids"], 2048)
    result = [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(value) for value in ids),
        "--device", "hip", "--top-k", "1", "--batch", "2",
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", "2112", "--new-tokens", "64",
        "--warmup", str(warmup), "--steps", str(steps),
        "--prefill-warmup", str(warmup), "--prefill-steps", str(steps),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true", "--bf16-ffn-arena-minimum-rows", "1",
        "--workload", "decode",
    ]
    if policy == "grouped65193":
        result += ["--bf16-grouped-gate-up-algorithm-index", str(SOLUTION)]
    if logits is not None:
        result += ["--cache-logits-output", str(logits), "--cache-logits-step", "0"]
    return result


def require(record: dict, policy: str, warmup: int, steps: int) -> None:
    expected_dispatches = 28 * 64 * (warmup + steps)
    expected = {
        "status": "pass", "batch": 2, "token_count": 2048,
        "decode_tokens": 64, "kv_cache_dtype": "bf16",
        "bf16_ffn_arena_enabled": True,
        "bf16_grouped_gate_up_algorithm_index":
            SOLUTION if policy == "grouped65193" else -1,
        "bf16_grouped_gate_up_registered_entries":
            1 if policy == "grouped65193" else 0,
        "bf16_grouped_gate_up_dispatches":
            expected_dispatches if policy == "grouped65193" else 0,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(f"{policy} {name} expected {wanted!r}, got {record.get(name)!r}")
    if policy == "grouped65193" and record.get(
            "bf16_grouped_gate_up_plan_entries") != 28:
        raise ValueError("grouped decode plan count changed")


def read_logits(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) != 2 * 151936 * 4:
        raise ValueError("grouped decode logits size changed")
    return list(struct.unpack(f"<{len(data)//4}f", data))


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    precision, performance = [], []
    with tempfile.TemporaryDirectory(prefix="microllm-grouped-decode-") as root:
        temp = Path(root)
        for run in range(1, 4):
            baseline_values = None
            for policy in POLICIES:
                path = temp / f"{policy}-{run}.bin"
                done = subprocess.run(command(args, model, policy, 0, 1, path),
                                      text=True, capture_output=True,
                                      timeout=args.timeout_seconds)
                if done.returncode: raise RuntimeError(done.stderr or done.stdout)
                record = COMMON.last_json(done.stdout); require(record, policy, 0, 1)
                values = read_logits(path)
                if baseline_values is None: baseline_values = values
                maximum, rms, bitwise = COMMON.error(baseline_values, values)
                precision.append({"policy": policy, "process_run": run,
                                  "maximum": maximum, "rms": rms,
                                  "bitwise_equal": bitwise,
                                  "generated_tokens": record["generated_tokens"]})
        for run in range(1, 4):
            order = list(POLICIES) if run % 2 else list(reversed(POLICIES))
            for policy in order:
                done = subprocess.run(command(args, model, policy, 2, 5),
                                      text=True, capture_output=True,
                                      timeout=args.timeout_seconds)
                if done.returncode: raise RuntimeError(done.stderr or done.stdout)
                record = COMMON.last_json(done.stdout); require(record, policy, 2, 5)
                performance.append({"policy": policy, "process_run": run,
                    "tokens_per_second": float(record["decode_tokens_per_second"]),
                    "peak_bytes": int(record["engine_peak_bytes"]),
                    "backend_allocations": int(record["engine_backend_allocation_calls"]),
                    "arena_capacity_bytes": int(record["bf16_ffn_arena_capacity_bytes"]),
                    "generated_tokens": record["generated_tokens"]})
    base = [r for r in performance if r["policy"] == "arena"]
    cand = [r for r in performance if r["policy"] == "grouped65193"]
    base_t = statistics.median(r["tokens_per_second"] for r in base)
    cand_t = statistics.median(r["tokens_per_second"] for r in cand)
    candidate_precision = [r for r in precision if r["policy"] == "grouped65193"]
    summary = {"schema_version": 1, "status": "pass",
        "record_type": "grouped_gate_up_decode_model_gate",
        "precision_process_rows": len(precision),
        "performance_process_rows": len(performance),
        "baseline_tokens_per_second": base_t,
        "candidate_tokens_per_second": cand_t,
        "throughput_speedup": cand_t / base_t,
        "maximum_logit_error": max(r["maximum"] for r in candidate_precision),
        "maximum_logit_rms": max(r["rms"] for r in candidate_precision),
        "tokens_equal": all(r["generated_tokens"] == precision[0]["generated_tokens"]
                            for r in precision),
        "peak_delta_bytes": max(r["peak_bytes"] for r in cand) - max(r["peak_bytes"] for r in base),
        "candidate_admitted": cand_t / base_t >= 1.01 and
            all(r["generated_tokens"] == precision[0]["generated_tokens"] for r in precision),
        "precision_rows": precision, "performance_rows": performance}
    for name, rows in (("precision-raw.jsonl", precision), ("performance-raw.jsonl", performance)):
        (args.output_directory/name).write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
    (args.output_directory/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith("_rows")},sort_keys=True))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"grouped_gate_up_decode_model_gate: {error}", file=sys.stderr); raise SystemExit(2)
