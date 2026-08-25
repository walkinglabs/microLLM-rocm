#!/usr/bin/env python3
"""Same-binary Model-S gate for gradient-ready communication overlap."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (
    ("transient", False, False, False),
    ("synchronous_views", True, True, False),
    ("overlap_views", True, True, True),
)
BUCKET_BYTES = 25 * 1024 * 1024


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0:
        parser.error("gradient overlap matrix inputs are invalid")
    return args


def execute(binary: Path, policy: str, persistent: bool, views: bool,
            overlap: bool, process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(binary), "--model", "model-s", "--steps", "5",
        "--context", "32", "--batch", "1",
        "--bucket-bytes", str(BUCKET_BYTES),
        "--parameter-check-interval", "5",
        "--persistent-gradient-buckets", "true" if persistent else "false",
        "--gradient-bucket-views", "true" if views else "false",
        "--overlap-gradient-communication", "true" if overlap else "false",
        "--seed", "601",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != 5:
        raise RuntimeError(f"{policy} did not emit five steps")
    for step, row in enumerate(rows, start=1):
        overlap_active = overlap and step > 1
        allocations = 120 if not persistent else (6 if step == 1 else 0)
        if (row.get("persistent_gradient_buckets") is not persistent or
                row.get("gradient_bucket_views") is not views or
                row.get("overlap_gradient_communication") is not overlap or
                row.get("overlap_communication_performed") is not overlap_active or
                row.get("bucket_overlap_enabled") is not overlap_active or
                row.get("overlapped_bucket_count") != (3 if overlap_active else 0) or
                row.get("bucket_count") != 3 or
                row.get("communication_allocation_calls") != allocations or
                row.get("communication_backend_allocation_calls") != allocations or
                row.get("communication_cache_reuse_calls") != 0 or
                row.get("unpacked_tensor_count") != (0 if views else 114) or
                row.get("gradient_view_count") != (114 if views else 0) or
                row.get("unpack_copy_calls") != (0 if views else 114) or
                row.get("bucket_plan_capacity_bytes") !=
                (124689408 if persistent else 0)):
            raise RuntimeError(f"{policy} overlap contract changed at step {step}")
    if (rows[-1].get("parameter_check_performed") is not True or
            rows[-1].get("parameter_max_difference") != 0.0):
        raise RuntimeError(f"{policy} parameter gate failed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_gradient_overlap_measurement",
            "policy": policy,
            "process_run": process_run,
        })
    return rows


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    raw = []
    processes = []
    for process_run in range(1, args.runs + 1):
        shift = (process_run - 1) % len(POLICIES)
        order = POLICIES[shift:] + POLICIES[:shift]
        loss_reference = None
        for policy, persistent, views, overlap in order:
            rows = execute(args.binary, policy, persistent, views,
                           overlap, process_run)
            losses = [float(row["mean_loss"]) for row in rows]
            if loss_reference is None:
                loss_reference = losses
            elif losses != loss_reference:
                raise RuntimeError("gradient overlap changed the loss trajectory")
            raw.extend(rows)
            steady = rows[1:]
            processes.append({
                "policy": policy,
                "process_run": process_run,
                "median_forward_backward_ms": median(steady, "forward_backward_ms"),
                "median_communication_ms": median(steady, "communication_ms"),
                "median_overlap_finish_ms": median(steady, "overlap_finish_ms"),
                "median_total_ms": median(steady, "total_ms"),
                "maximum_engine_peak_bytes": max(
                    int(row["maximum_engine_peak_bytes"]) for row in steady),
                "median_engine_current_bytes": int(
                    median(steady, "maximum_engine_current_bytes")),
                "final_loss": losses[-1],
            })
    policies = {}
    for policy, persistent, views, overlap in POLICIES:
        rows = [row for row in processes if row["policy"] == policy]
        policies[policy] = {
            "persistent": persistent,
            "views": views,
            "overlap": overlap,
            "processes": len(rows),
            "median_forward_backward_ms": median(rows, "median_forward_backward_ms"),
            "median_communication_ms": median(rows, "median_communication_ms"),
            "median_overlap_finish_ms": median(rows, "median_overlap_finish_ms"),
            "median_total_ms": median(rows, "median_total_ms"),
            "maximum_engine_peak_bytes": int(
                median(rows, "maximum_engine_peak_bytes")),
            "median_engine_current_bytes": int(
                median(rows, "median_engine_current_bytes")),
            "final_loss": median(rows, "final_loss"),
        }
    transient = policies["transient"]
    synchronous = policies["synchronous_views"]
    overlap = policies["overlap_views"]
    speedup_vs_sync = synchronous["median_total_ms"] / overlap["median_total_ms"]
    speedup_vs_transient = transient["median_total_ms"] / overlap["median_total_ms"]
    peak_delta_sync = (overlap["maximum_engine_peak_bytes"] -
                       synchronous["maximum_engine_peak_bytes"])
    peak_delta_transient = (overlap["maximum_engine_peak_bytes"] -
                            transient["maximum_engine_peak_bytes"])
    retained = speedup_vs_sync >= 1.01 and peak_delta_sync <= 0
    default_eligible = retained and peak_delta_transient <= 0
    if default_eligible:
        decision = "keep gradient overlap as default"
    elif retained:
        decision = "keep explicit and move to one-process-per-GPU"
    else:
        decision = "reject single-process gradient overlap route"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_gradient_overlap_summary",
        "raw_records": len(raw),
        "processes": len(processes),
        "runs_per_policy": args.runs,
        "aggregation": "median of step-2..5 medians with rotated policy order",
        "loss_trajectories_exact": True,
        "policies": policies,
        "total_speedup_vs_synchronous_views": speedup_vs_sync,
        "total_speedup_vs_transient": speedup_vs_transient,
        "finish_wait_speedup_vs_synchronous_communication": (
            synchronous["median_communication_ms"] /
            overlap["median_overlap_finish_ms"]),
        "peak_bytes_added_vs_synchronous_views": peak_delta_sync,
        "peak_bytes_added_vs_transient": peak_delta_transient,
        "retained": retained,
        "default_eligible": default_eligible,
        "decision": decision,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
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
        print(f"data_parallel_gradient_overlap_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
