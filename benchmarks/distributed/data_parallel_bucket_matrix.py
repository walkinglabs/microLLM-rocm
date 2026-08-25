#!/usr/bin/env python3
"""Measure real tiny-model bucket counts and communication cost."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (("4b", 4), ("64b", 64), ("4kib", 4096), ("4mib", 4 * 1024 * 1024))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=601)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.steps != 20 or args.seed < 0:
        parser.error("bucket matrix requires pinned 20-step inputs")
    return args


def execute(args: argparse.Namespace, policy: str, bucket_bytes: int,
            process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(args.binary), "--steps", str(args.steps),
        "--bucket-bytes", str(bucket_bytes),
        "--parameter-check-interval", str(args.steps),
        "--seed", str(args.seed),
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    bucket_counts = {int(row.get("bucket_count", 0)) for row in rows}
    if (len(rows) != args.steps or len(bucket_counts) != 1 or
            next(iter(bucket_counts)) <= 0 or
            sum(bool(row.get("parameter_check_performed")) for row in rows) != 1 or
            rows[-1].get("parameter_check_performed") is not True or
            any(float(row.get("parameter_max_difference", 0.0)) != 0.0 for row in rows)):
        raise RuntimeError(f"{policy} bucket/check contract changed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_bucket_measurement",
            "policy": policy, "bucket_bytes": bucket_bytes,
            "process_run": process_run,
        })
    return rows


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    records = []
    process_summaries = []
    for process_run in range(1, args.runs + 1):
        rotation = (process_run - 1) % len(POLICIES)
        order = POLICIES[rotation:] + POLICIES[:rotation]
        loss_reference = None
        for policy, bucket_bytes in order:
            rows = execute(args, policy, bucket_bytes, process_run)
            losses = [float(row["mean_loss"]) for row in rows]
            if loss_reference is None:
                loss_reference = losses
            elif losses != loss_reference:
                raise RuntimeError("bucket size changed the loss trajectory")
            records.extend(rows)
            steady = rows[1:]
            process_summaries.append({
                "policy": policy,
                "bucket_bytes": bucket_bytes,
                "process_run": process_run,
                "bucket_count": int(steady[0]["bucket_count"]),
                "median_forward_backward_ms": median(steady, "forward_backward_ms"),
                "median_communication_ms": median(steady, "communication_ms"),
                "median_optimizer_ms": median(steady, "optimizer_ms"),
                "median_total_ms": median(steady, "total_ms"),
                "final_loss": losses[-1],
            })
    policies = {}
    for policy, bucket_bytes in POLICIES:
        rows = [row for row in process_summaries if row["policy"] == policy]
        policies[policy] = {
            "bucket_bytes": bucket_bytes,
            "bucket_count": int(median(rows, "bucket_count")),
            "processes": len(rows),
            "median_forward_backward_ms": median(rows, "median_forward_backward_ms"),
            "median_communication_ms": median(rows, "median_communication_ms"),
            "median_optimizer_ms": median(rows, "median_optimizer_ms"),
            "median_total_ms": median(rows, "median_total_ms"),
            "final_loss": median(rows, "final_loss"),
        }
    reference = policies["4mib"]["median_total_ms"]
    for policy in policies:
        policies[policy]["speedup_vs_4mib"] = (
            reference / policies[policy]["median_total_ms"])
    one_bucket_equivalent = (
        policies["4kib"]["bucket_count"] == 1 and
        policies["4mib"]["bucket_count"] == 1)
    multi_bucket_slower = all(
        policies[policy]["median_total_ms"] >
        max(policies["4kib"]["median_total_ms"],
            policies["4mib"]["median_total_ms"])
        for policy in ("4b", "64b"))
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_bucket_matrix_summary",
        "raw_records": len(records),
        "processes": len(process_summaries),
        "runs_per_policy": args.runs,
        "steps_per_process": args.steps,
        "steady_steps_per_process": args.steps - 1,
        "aggregation": "median of step-2..20 medians with rotated bucket order",
        "loss_trajectories_exact": True,
        "policies": policies,
        "one_bucket_policies_are_equivalent_workloads": one_bucket_equivalent,
        "multi_bucket_policies_are_slower": multi_bucket_slower,
        "decision": ("add a Model-S multi-bucket workload before overlap work"
                     if one_bucket_equivalent and multi_bucket_slower else
                     "investigate the measured tiny-model bucket matrix"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "process-summary.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in process_summaries), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"data_parallel_bucket_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
