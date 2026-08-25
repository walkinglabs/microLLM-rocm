#!/usr/bin/env python3
"""Same-binary Model-S gate for persistent gradient-bucket Storage."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (("transient", False), ("persistent", True))
EXPECTED_PLAN_BYTES = 249378816


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0:
        parser.error("persistent bucket matrix inputs are invalid")
    return args


def execute(binary: Path, policy: str, enabled: bool,
            process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(binary), "--model", "model-s", "--steps", "5",
        "--context", "32", "--batch", "1", "--bucket-bytes", "26214400",
        "--parameter-check-interval", "5", "--inplace-bucket-average", "true",
        "--persistent-gradient-buckets", "true" if enabled else "false",
        "--seed", "601",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != 5:
        raise RuntimeError(f"{policy} did not emit five steps")
    for step, row in enumerate(rows, start=1):
        expected_allocations = 120 if not enabled or step == 1 else 0
        expected_bytes = EXPECTED_PLAN_BYTES if expected_allocations else 0
        expected_reused = enabled and step > 1
        if (row.get("persistent_gradient_buckets") is not enabled or
                row.get("bucket_persistent_storage") is not enabled or
                row.get("bucket_plan_reused") is not expected_reused or
                row.get("bucket_count") != 3 or
                row.get("bucket_tensor_count") != 6 or
                row.get("average_tensor_count") != 0 or
                row.get("unpacked_tensor_count") != 114 or
                row.get("pack_copy_calls") != 114 or
                row.get("unpack_copy_calls") != 114 or
                row.get("communication_allocation_calls") != expected_allocations or
                row.get("communication_backend_allocation_calls") !=
                expected_allocations or
                row.get("communication_cache_reuse_calls") != 0 or
                row.get("communication_total_allocated_bytes") != expected_bytes or
                row.get("bucket_plan_capacity_bytes") !=
                (EXPECTED_PLAN_BYTES if enabled else 0) or
                row.get("bucket_temporary_bytes") !=
                (0 if enabled else EXPECTED_PLAN_BYTES)):
            raise RuntimeError(f"{policy} persistent bucket contract changed at step {step}")
    if (rows[-1].get("parameter_check_performed") is not True or
            rows[-1].get("parameter_max_difference") != 0.0):
        raise RuntimeError(f"{policy} parameter gate failed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_persistent_bucket_measurement",
            "policy": policy,
            "process_run": process_run,
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
                raise RuntimeError("persistent buckets changed the loss trajectory")
            records.extend(rows)
            steady = rows[1:]
            processes.append({
                "policy": policy,
                "process_run": process_run,
                "median_communication_ms": median(steady, "communication_ms"),
                "median_total_ms": median(steady, "total_ms"),
                "maximum_engine_peak_bytes": max(
                    int(row["maximum_engine_peak_bytes"]) for row in steady),
                "median_engine_current_bytes": int(
                    median(steady, "maximum_engine_current_bytes")),
                "final_loss": losses[-1],
            })
    policies = {}
    for policy, enabled in POLICIES:
        rows = [row for row in processes if row["policy"] == policy]
        policies[policy] = {
            "enabled": enabled,
            "processes": len(rows),
            "median_communication_ms": median(rows, "median_communication_ms"),
            "median_total_ms": median(rows, "median_total_ms"),
            "maximum_engine_peak_bytes": int(
                median(rows, "maximum_engine_peak_bytes")),
            "median_engine_current_bytes": int(
                median(rows, "median_engine_current_bytes")),
            "final_loss": median(rows, "final_loss"),
        }
    baseline = policies["transient"]
    candidate = policies["persistent"]
    total_speedup = baseline["median_total_ms"] / candidate["median_total_ms"]
    peak_bytes_added = (candidate["maximum_engine_peak_bytes"] -
                        baseline["maximum_engine_peak_bytes"])
    current_bytes_added = (candidate["median_engine_current_bytes"] -
                           baseline["median_engine_current_bytes"])
    allocation_gate = all(
        row["communication_backend_allocation_calls"] == 0
        for row in records
        if row["policy"] == "persistent" and row["step"] > 1)
    memory_gate = (peak_bytes_added <= EXPECTED_PLAN_BYTES and
                   current_bytes_added <= EXPECTED_PLAN_BYTES)
    eligible_for_default = (allocation_gate and total_speedup >= 1.01 and
                            peak_bytes_added <= 0 and current_bytes_added <= 0)
    retained = allocation_gate and total_speedup >= 1.01 and memory_gate
    if eligible_for_default:
        decision = "keep persistent gradient buckets as default"
    elif retained:
        decision = "keep explicit and continue to view-backed gradients"
    else:
        decision = "reject persistent gradient bucket model route"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_persistent_bucket_summary",
        "raw_records": len(records),
        "processes": len(processes),
        "runs_per_policy": args.runs,
        "aggregation": "median of step-2..5 medians with alternating policy order",
        "loss_trajectories_exact": True,
        "later_step_backend_allocations_eliminated": allocation_gate,
        "plan_capacity_bytes": EXPECTED_PLAN_BYTES,
        "policies": policies,
        "total_speedup": total_speedup,
        "communication_speedup": (
            baseline["median_communication_ms"] /
            candidate["median_communication_ms"]),
        "peak_bytes_added": peak_bytes_added,
        "current_bytes_added": current_bytes_added,
        "memory_gate": memory_gate,
        "eligible_for_default": eligible_for_default,
        "decision": decision,
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
    except (OSError, RuntimeError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"data_parallel_persistent_bucket_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
