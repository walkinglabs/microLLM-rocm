#!/usr/bin/env python3
"""Same-binary Model-S gate for direct Autograd accumulation into buckets."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (
    ("transient", False, False, False),
    ("bucket_views", True, True, False),
    ("direct", True, True, True),
)
TRANSIENT_BYTES = 249378816
BUCKET_BYTES = 124689408


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0:
        parser.error("direct bucket gradient matrix inputs are invalid")
    return args


def execute(binary: Path, policy: str, persistent: bool, views: bool,
            direct: bool, process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(binary), "--model", "model-s", "--steps", "5",
        "--context", "32", "--batch", "1", "--bucket-bytes", "26214400",
        "--parameter-check-interval", "5", "--inplace-bucket-average", "true",
        "--persistent-gradient-buckets", "true" if persistent else "false",
        "--gradient-bucket-views", "true" if views else "false",
        "--direct-bucket-gradients", "true" if direct else "false",
        "--seed", "601",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != 5:
        raise RuntimeError(f"{policy} did not emit five steps")
    for step, row in enumerate(rows, start=1):
        initialized = persistent and step > 1
        allocations = 120 if not persistent else (6 if step == 1 else 0)
        allocated_bytes = (TRANSIENT_BYTES if not persistent else
                           BUCKET_BYTES if step == 1 else 0)
        expected_pack = 0 if direct and initialized else 114
        expected_direct = 114 if direct and initialized else 0
        expected_unpacked = 114 if not views else 0
        expected_views = 114 if views else 0
        expected_unpack = 114 if not views else 0
        if (row.get("persistent_gradient_buckets") is not persistent or
                row.get("gradient_bucket_views") is not views or
                row.get("direct_bucket_gradients") is not direct or
                row.get("bucket_plan_reused") is not initialized or
                row.get("bucket_count") != 3 or
                row.get("unpacked_tensor_count") != expected_unpacked or
                row.get("gradient_view_count") != expected_views or
                row.get("direct_gradient_target_count") != expected_direct or
                row.get("pack_copy_calls") != expected_pack or
                row.get("unpack_copy_calls") != expected_unpack or
                row.get("communication_allocation_calls") != allocations or
                row.get("communication_backend_allocation_calls") != allocations or
                row.get("communication_cache_reuse_calls") != 0 or
                row.get("communication_total_allocated_bytes") != allocated_bytes or
                row.get("bucket_plan_capacity_bytes") !=
                (BUCKET_BYTES if persistent else 0)):
            raise RuntimeError(f"{policy} direct gradient contract changed at step {step}")
    if (rows[-1].get("parameter_check_performed") is not True or
            rows[-1].get("parameter_max_difference") != 0.0):
        raise RuntimeError(f"{policy} parameter gate failed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_direct_bucket_gradient_measurement",
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
        for policy, persistent, views, direct in order:
            rows = execute(args.binary, policy, persistent, views, direct, process_run)
            losses = [float(row["mean_loss"]) for row in rows]
            if loss_reference is None:
                loss_reference = losses
            elif losses != loss_reference:
                raise RuntimeError("direct bucket gradients changed the loss trajectory")
            records.extend(rows)
            steady = rows[1:]
            processes.append({
                "policy": policy,
                "process_run": process_run,
                "median_forward_backward_ms": median(steady, "forward_backward_ms"),
                "median_communication_ms": median(steady, "communication_ms"),
                "median_total_ms": median(steady, "total_ms"),
                "maximum_engine_peak_bytes": max(
                    int(row["maximum_engine_peak_bytes"]) for row in steady),
                "median_engine_current_bytes": int(
                    median(steady, "maximum_engine_current_bytes")),
                "final_loss": losses[-1],
            })
    policies = {}
    for policy, persistent, views, direct in POLICIES:
        rows = [row for row in processes if row["policy"] == policy]
        policies[policy] = {
            "persistent": persistent,
            "views": views,
            "direct": direct,
            "processes": len(rows),
            "median_forward_backward_ms": median(rows, "median_forward_backward_ms"),
            "median_communication_ms": median(rows, "median_communication_ms"),
            "median_total_ms": median(rows, "median_total_ms"),
            "maximum_engine_peak_bytes": int(
                median(rows, "maximum_engine_peak_bytes")),
            "median_engine_current_bytes": int(
                median(rows, "median_engine_current_bytes")),
            "final_loss": median(rows, "final_loss"),
        }
    transient = policies["transient"]
    views = policies["bucket_views"]
    direct = policies["direct"]
    total_speedup_vs_views = views["median_total_ms"] / direct["median_total_ms"]
    total_speedup_vs_transient = transient["median_total_ms"] / direct["median_total_ms"]
    peak_added_vs_transient = (direct["maximum_engine_peak_bytes"] -
                               transient["maximum_engine_peak_bytes"])
    default_eligible = (total_speedup_vs_transient >= 1.01 and
                        direct["median_engine_current_bytes"] <=
                        transient["median_engine_current_bytes"] and
                        peak_added_vs_transient <= 0)
    retained = (total_speedup_vs_views >= 1.01 and
                direct["maximum_engine_peak_bytes"] <=
                views["maximum_engine_peak_bytes"])
    if default_eligible:
        decision = "keep direct bucket gradients as default"
    elif retained:
        decision = "keep explicit and require producer out-kernels before default"
    else:
        decision = "reject direct bucket-gradient model route"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_direct_bucket_gradient_summary",
        "raw_records": len(records),
        "processes": len(processes),
        "runs_per_policy": args.runs,
        "aggregation": "median of step-2..5 medians with rotated policy order",
        "loss_trajectories_exact": True,
        "policies": policies,
        "forward_backward_speedup_vs_views": (
            views["median_forward_backward_ms"] /
            direct["median_forward_backward_ms"]),
        "communication_speedup_vs_views": (
            views["median_communication_ms"] / direct["median_communication_ms"]),
        "total_speedup_vs_views": total_speedup_vs_views,
        "total_speedup_vs_transient": total_speedup_vs_transient,
        "peak_bytes_saved_vs_views": (
            views["maximum_engine_peak_bytes"] -
            direct["maximum_engine_peak_bytes"]),
        "peak_bytes_added_vs_transient": peak_added_vs_transient,
        "pack_copies_removed": 114,
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
        print(f"data_parallel_direct_bucket_gradient_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
