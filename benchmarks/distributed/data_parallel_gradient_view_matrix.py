#!/usr/bin/env python3
"""Same-binary Model-S gate for gradient views into reduced buckets."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (
    ("transient", False, False),
    ("persistent_copy", True, False),
    ("bucket_views", True, True),
)
FULL_GRADIENT_BYTES = 249378816
BUCKET_ONLY_BYTES = 124689408


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0:
        parser.error("gradient view matrix inputs are invalid")
    return args


def execute(binary: Path, policy: str, persistent: bool, views: bool,
            process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(binary), "--model", "model-s", "--steps", "5",
        "--context", "32", "--batch", "1", "--bucket-bytes", "26214400",
        "--parameter-check-interval", "5", "--inplace-bucket-average", "true",
        "--persistent-gradient-buckets", "true" if persistent else "false",
        "--gradient-bucket-views", "true" if views else "false",
        "--seed", "601",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != 5:
        raise RuntimeError(f"{policy} did not emit five steps")
    for step, row in enumerate(rows, start=1):
        if not persistent:
            allocations = 120
            allocated_bytes = FULL_GRADIENT_BYTES
            capacity = 0
            temporary = FULL_GRADIENT_BYTES
        elif step == 1:
            allocations = 6 if views else 120
            allocated_bytes = BUCKET_ONLY_BYTES if views else FULL_GRADIENT_BYTES
            capacity = allocated_bytes
            temporary = 0
        else:
            allocations = 0
            allocated_bytes = 0
            capacity = BUCKET_ONLY_BYTES if views else FULL_GRADIENT_BYTES
            temporary = 0
        expected_unpacked = 0 if views else 114
        expected_views = 114 if views else 0
        expected_unpack_copies = 0 if views else 114
        if (row.get("persistent_gradient_buckets") is not persistent or
                row.get("gradient_bucket_views") is not views or
                row.get("bucket_persistent_storage") is not persistent or
                row.get("bucket_plan_reused") is not (persistent and step > 1) or
                row.get("bucket_count") != 3 or
                row.get("bucket_tensor_count") != 6 or
                row.get("unpacked_tensor_count") != expected_unpacked or
                row.get("gradient_view_count") != expected_views or
                row.get("pack_copy_calls") != 114 or
                row.get("unpack_copy_calls") != expected_unpack_copies or
                row.get("communication_allocation_calls") != allocations or
                row.get("communication_backend_allocation_calls") != allocations or
                row.get("communication_cache_reuse_calls") != 0 or
                row.get("communication_total_allocated_bytes") != allocated_bytes or
                row.get("bucket_plan_capacity_bytes") != capacity or
                row.get("bucket_temporary_bytes") != temporary):
            raise RuntimeError(f"{policy} gradient view contract changed at step {step}")
    if (rows[-1].get("parameter_check_performed") is not True or
            rows[-1].get("parameter_max_difference") != 0.0):
        raise RuntimeError(f"{policy} parameter gate failed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_gradient_view_measurement",
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
        shift = (process_run - 1) % len(POLICIES)
        order = POLICIES[shift:] + POLICIES[:shift]
        loss_reference = None
        for policy, persistent, views in order:
            rows = execute(args.binary, policy, persistent, views, process_run)
            losses = [float(row["mean_loss"]) for row in rows]
            if loss_reference is None:
                loss_reference = losses
            elif losses != loss_reference:
                raise RuntimeError("gradient bucket views changed the loss trajectory")
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
    for policy, persistent, views in POLICIES:
        rows = [row for row in processes if row["policy"] == policy]
        policies[policy] = {
            "persistent": persistent,
            "views": views,
            "processes": len(rows),
            "median_communication_ms": median(rows, "median_communication_ms"),
            "median_total_ms": median(rows, "median_total_ms"),
            "maximum_engine_peak_bytes": int(
                median(rows, "maximum_engine_peak_bytes")),
            "median_engine_current_bytes": int(
                median(rows, "median_engine_current_bytes")),
            "final_loss": median(rows, "final_loss"),
        }
    transient = policies["transient"]
    copied = policies["persistent_copy"]
    views = policies["bucket_views"]
    view_vs_copy_total = copied["median_total_ms"] / views["median_total_ms"]
    view_vs_transient_total = transient["median_total_ms"] / views["median_total_ms"]
    current_vs_transient = (views["median_engine_current_bytes"] -
                            transient["median_engine_current_bytes"])
    peak_vs_transient = (views["maximum_engine_peak_bytes"] -
                         transient["maximum_engine_peak_bytes"])
    storage_gate = (copied["median_engine_current_bytes"] -
                    views["median_engine_current_bytes"] == BUCKET_ONLY_BYTES)
    default_eligible = (view_vs_transient_total >= 1.01 and
                        current_vs_transient <= 0 and peak_vs_transient <= 0)
    retained = (storage_gate and view_vs_copy_total >= 1.01 and
                view_vs_transient_total >= 1.01)
    if default_eligible:
        decision = "keep persistent bucket views as default"
    elif retained:
        decision = "keep explicit and continue to direct bucket-gradient accumulation"
    else:
        decision = "reject gradient bucket view model route"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_gradient_view_summary",
        "raw_records": len(records),
        "processes": len(processes),
        "runs_per_policy": args.runs,
        "aggregation": "median of step-2..5 medians with rotated policy order",
        "loss_trajectories_exact": True,
        "policies": policies,
        "view_vs_copy_communication_speedup": (
            copied["median_communication_ms"] / views["median_communication_ms"]),
        "view_vs_copy_total_speedup": view_vs_copy_total,
        "view_vs_transient_communication_speedup": (
            transient["median_communication_ms"] / views["median_communication_ms"]),
        "view_vs_transient_total_speedup": view_vs_transient_total,
        "current_bytes_saved_vs_copy": (
            copied["median_engine_current_bytes"] -
            views["median_engine_current_bytes"]),
        "peak_bytes_saved_vs_copy": (
            copied["maximum_engine_peak_bytes"] -
            views["maximum_engine_peak_bytes"]),
        "current_bytes_added_vs_transient": current_vs_transient,
        "peak_bytes_added_vs_transient": peak_vs_transient,
        "unpack_storage_removed": 114,
        "unpack_copies_removed": 114,
        "plan_capacity_bytes": BUCKET_ONLY_BYTES,
        "default_eligible": default_eligible,
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
        print(f"data_parallel_gradient_view_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
