#!/usr/bin/env python3
"""Three-process short-decode gate for the Qwen3 down-FP32 candidate."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


POLICIES = ("current", "down-fp32")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    for path in (args.config, args.weights, args.binary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, policy: str) -> list[str]:
    result = [
        str(args.binary), "--config", str(args.config), "--weights", str(args.weights),
        "--tokens", "1", "--device", "hip", "--top-k", "1", "--batch", "1",
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", "5", "--new-tokens", "4",
        "--warmup", "2", "--steps", "5", "--prefill-warmup", "2",
        "--prefill-steps", "5", "--bf16-ffn", "true",
        "--bf16-attention", "true", "--workload", "decode",
    ]
    if policy == "down-fp32":
        result.extend(["--bf16-ffn-weight-scope", "gate-up"])
    return result


def run_one(args: argparse.Namespace, policy: str,
            process_run: int, order: tuple[str, ...]) -> dict:
    completed = subprocess.run(
        command(args, policy), capture_output=True, text=True,
        timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    source = json.loads(completed.stdout.strip())
    expected_scope = "gate-up" if policy == "down-fp32" else "all"
    expected_ffn = 56 if policy == "down-fp32" else 84
    if (source.get("status") != "pass" or source.get("warmup") != 2 or
            source.get("steps") != 5 or source.get("decode_tokens") != 4 or
            source.get("measured_forward_steps") != 20 or
            source.get("bf16_ffn_weight_scope") != expected_scope or
            source.get("bf16_ffn_converted_tensors") != expected_ffn or
            source.get("bf16_attention_converted_tensors") != 112 or
            source.get("generated_tokens") != [25, 16246, 264, 738]):
        raise RuntimeError(f"{policy} changed the formal short-decode contract")
    return {
        "schema_version": 1, "record_type": "qwen3_down_fp32_short_sample",
        "status": "pass", "policy": policy, "process_run": process_run,
        "pair_order": list(order), "warmup": 2, "steps": 5,
        "decode_tokens": 4,
        "throughput_tokens_per_second": float(source["decode_tokens_per_second"]),
        "latency_ms": float(source["mean_generation_ms"]),
        "resident_weight_bytes": int(source["resident_weight_bytes"]),
        "engine_peak_bytes": int(source["engine_peak_bytes"]),
        "preparation_peak_bytes": int(source["preparation_peak_bytes"]),
        "generated_tokens": source["generated_tokens"],
    }


def summarize(records: list[dict]) -> dict:
    by_policy = {policy: [row for row in records if row["policy"] == policy]
                 for policy in POLICIES}
    if any(len(rows) != 3 for rows in by_policy.values()):
        raise RuntimeError("formal short gate requires three samples per policy")
    medians = {}
    for policy, rows in by_policy.items():
        for field in ("throughput_tokens_per_second", "latency_ms",
                      "resident_weight_bytes", "engine_peak_bytes",
                      "preparation_peak_bytes"):
            medians[f"{policy}_{field}"] = statistics.median(
                row[field] for row in rows)
    ratio = (medians["down-fp32_throughput_tokens_per_second"] /
             medians["current_throughput_tokens_per_second"])
    latency_ratio = medians["down-fp32_latency_ms"] / medians["current_latency_ms"]
    gates = {
        "throughput_at_least_point95": ratio >= 0.95,
        "latency_at_most_1_05": latency_ratio <= 1.05,
        "resident_delta_exact":
            medians["down-fp32_resident_weight_bytes"] -
            medians["current_resident_weight_bytes"] == 176_160_768,
        "tokens_equal": by_policy["current"][0]["generated_tokens"] ==
            by_policy["down-fp32"][0]["generated_tokens"],
    }
    return {
        "schema_version": 1, "record_type": "qwen3_down_fp32_short_gate",
        "status": "pass_performance" if all(gates.values()) else "reject_performance",
        "case": "T1/B1/N4 cached decode", "runs": 3, "warmup": 2, "steps": 5,
        **medians, "candidate_over_current_throughput": ratio,
        "candidate_over_current_latency": latency_ratio,
        "gates": gates,
        "boundary": "short-decode early-stop gate; complete shape was intentionally stopped",
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in (1, 2, 3):
        order = POLICIES if process_run % 2 else tuple(reversed(POLICIES))
        for policy in order:
            record = run_one(args, policy, process_run, order)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize(records)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"].startswith("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
