#!/usr/bin/env python3
"""Run the fixed MI300 BF16 FFN matrix and preserve every process result."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path


PROFILES = (
    ("qwen", 896, 4864),
    ("deepseek", 1536, 8960),
)
TOKENS = (1, 128)
PATHS = ("fp32", "per-linear", "island")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    result = parser.parse_args()
    if result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs/repetitions must be positive and warmup non-negative")
    if not result.benchmark.is_file():
        parser.error(f"benchmark does not exist: {result.benchmark}")
    return result


def median(records, key):
    return statistics.median(record[key] for record in records)


def main():
    options = arguments()
    records = []
    for model, hidden, intermediate in PROFILES:
        for tokens in TOKENS:
            for path in PATHS:
                for run in range(1, options.runs + 1):
                    command = [
                        str(options.benchmark),
                        "--path", path,
                        "--tokens", str(tokens),
                        "--hidden", str(hidden),
                        "--intermediate", str(intermediate),
                        "--warmup", str(options.warmup),
                        "--repetitions", str(options.repetitions),
                    ]
                    completed = subprocess.run(
                        command, check=True, text=True, capture_output=True
                    )
                    record = json.loads(completed.stdout)
                    record.update({"model": model, "process_run": run})
                    records.append(record)
                    print(json.dumps(record, sort_keys=True))

    groups = {}
    for record in records:
        key = (record["model"], record["tokens"], record["path"])
        groups.setdefault(key, []).append(record)
    rows = []
    for model, _, _ in PROFILES:
        for tokens in TOKENS:
            paths = {}
            for path in PATHS:
                group = groups[(model, tokens, path)]
                paths[path] = {
                    "device_ms_median_of_processes": median(group, "device_ms_median"),
                    "wall_ms_median_of_processes": median(group, "wall_ms_median"),
                    "max_abs_error_vs_fp32": max(r["max_abs_error_vs_fp32"] for r in group),
                    "relative_l2_error_vs_fp32": max(
                        r["relative_l2_error_vs_fp32"] for r in group
                    ),
                    "peak_active_bytes_median": median(group, "peak_active_bytes"),
                    "all_accuracy_passed": all(r["accuracy_passed"] for r in group),
                    "all_payload_transfers_zero": all(
                        r["host_to_device_calls_measured"] == 0
                        and r["device_to_host_calls_measured"] == 0
                        for r in group
                    ),
                }
            fp32 = paths["fp32"]["device_ms_median_of_processes"]
            per_linear = paths["per-linear"]["device_ms_median_of_processes"]
            island = paths["island"]["device_ms_median_of_processes"]
            rows.append({
                "model": model,
                "tokens": tokens,
                "paths": paths,
                "island_speedup_vs_fp32": fp32 / island,
                "island_speedup_vs_per_linear": per_linear / island,
            })
    summary = {
        "schema_version": 1,
        "track": "bf16_ffn",
        "aggregation": "median of three independent process medians by default",
        "rows": rows,
    }
    options.raw_output.parent.mkdir(parents=True, exist_ok=True)
    options.summary_output.parent.mkdir(parents=True, exist_ok=True)
    options.raw_output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    options.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
