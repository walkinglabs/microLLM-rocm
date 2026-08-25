#!/usr/bin/env python3
"""Same-binary Model-S gate for in-place bucket averaging."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (("allocating", False), ("inplace", True))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0:
        parser.error("in-place average matrix inputs are invalid")
    return args


def execute(binary: Path, policy: str, enabled: bool, process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(binary), "--model", "model-s", "--steps", "5",
        "--context", "32", "--batch", "1", "--bucket-bytes", "26214400",
        "--parameter-check-interval", "5", "--inplace-bucket-average",
        "true" if enabled else "false", "--seed", "601",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    expected_average = 0 if enabled else 6
    expected_allocations = 120 if enabled else 126
    expected_bytes = 249378816 if enabled else 374068224
    if (len(rows) != 5 or any(
            row.get("inplace_bucket_average") is not enabled or
            row.get("bucket_count") != 3 or
            row.get("average_tensor_count") != expected_average or
            row.get("communication_allocation_calls") != expected_allocations or
            row.get("communication_backend_allocation_calls") != expected_allocations or
            row.get("communication_total_allocated_bytes") != expected_bytes
            for row in rows) or
            rows[-1].get("parameter_check_performed") is not True or
            rows[-1].get("parameter_max_difference") != 0.0):
        raise RuntimeError(f"{policy} average contract changed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_inplace_average_measurement",
            "policy": policy, "process_run": process_run,
        })
    return rows


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    records = []
    processes = []
    for process_run in range(1, args.runs + 1):
        order = POLICIES if process_run % 2 else tuple(reversed(POLICIES))
        loss_reference = None
        for policy, enabled in order:
            rows = execute(args.binary, policy, enabled, process_run)
            losses = [float(row["mean_loss"]) for row in rows]
            if loss_reference is None:
                loss_reference = losses
            elif losses != loss_reference:
                raise RuntimeError("in-place average changed the loss trajectory")
            records.extend(rows)
            steady = rows[1:]
            processes.append({
                "policy": policy, "process_run": process_run,
                "median_communication_ms": median(steady, "communication_ms"),
                "median_total_ms": median(steady, "total_ms"),
                "maximum_engine_peak_bytes": int(steady[0]["maximum_engine_peak_bytes"]),
                "final_loss": losses[-1],
            })
    policies = {}
    for policy, enabled in POLICIES:
        rows = [row for row in processes if row["policy"] == policy]
        policies[policy] = {
            "enabled": enabled, "processes": len(rows),
            "median_communication_ms": median(rows, "median_communication_ms"),
            "median_total_ms": median(rows, "median_total_ms"),
            "maximum_engine_peak_bytes": int(median(rows, "maximum_engine_peak_bytes")),
            "final_loss": median(rows, "final_loss"),
        }
    baseline, candidate = policies["allocating"], policies["inplace"]
    speedup = baseline["median_total_ms"] / candidate["median_total_ms"]
    accepted = (speedup >= 1.01 and
                candidate["maximum_engine_peak_bytes"] <=
                baseline["maximum_engine_peak_bytes"])
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "data_parallel_inplace_average_summary",
        "raw_records": len(records), "processes": len(processes),
        "runs_per_policy": args.runs,
        "aggregation": "median of step-2..5 medians with alternating policy order",
        "loss_trajectories_exact": True, "policies": policies,
        "total_speedup": speedup,
        "communication_speedup": baseline["median_communication_ms"] /
            candidate["median_communication_ms"],
        "peak_bytes_saved": baseline["maximum_engine_peak_bytes"] -
            candidate["maximum_engine_peak_bytes"],
        "decision": ("keep in-place bucket average as default"
                     if accepted else "reject default in-place bucket average"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "process-summary.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"data_parallel_inplace_average_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)

