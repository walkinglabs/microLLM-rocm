#!/usr/bin/env python3
"""Measure every-step, sparse, and disabled data-parallel parameter audits."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (("every_step", 1), ("final_step", 20), ("disabled", 0))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bucket-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=601)
    args = parser.parse_args()
    if (not args.binary.is_file() or args.runs <= 0 or args.steps != 20 or
            args.bucket_bytes < 4 or args.seed < 0):
        parser.error("verification matrix requires the pinned 20-step inputs")
    return args


def execute(args: argparse.Namespace, policy: str, interval: int,
            process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(args.binary), "--steps", str(args.steps),
        "--bucket-bytes", str(args.bucket_bytes),
        "--parameter-check-interval", str(interval),
        "--seed", str(args.seed),
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != args.steps:
        raise RuntimeError(f"{policy} emitted an incomplete trajectory")
    expected_checks = args.steps if interval == 1 else 1 if interval == args.steps else 0
    if (sum(bool(row.get("parameter_check_performed")) for row in rows) !=
            expected_checks or
            any((not row.get("parameter_check_performed") and
                 float(row.get("verification_ms", -1.0)) != 0.0) or
                float(row.get("parameter_max_difference", 0.0)) != 0.0
                for row in rows)):
        raise RuntimeError(f"{policy} verification semantics changed")
    for row in rows:
        row.update({
            "record_type": "data_parallel_verification_measurement",
            "policy": policy,
            "parameter_check_interval": interval,
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
        for policy, interval in order:
            rows = execute(args, policy, interval, process_run)
            losses = [float(row["mean_loss"]) for row in rows]
            if loss_reference is None:
                loss_reference = losses
            elif losses != loss_reference:
                raise RuntimeError("parameter audit interval changed the loss trajectory")
            records.extend(rows)
            steady = rows[1:]
            process_summaries.append({
                "policy": policy,
                "parameter_check_interval": interval,
                "process_run": process_run,
                "steady_steps": len(steady),
                "median_total_ms": median(steady, "total_ms"),
                "median_verification_ms": median(
                    [row for row in rows if row["parameter_check_performed"]],
                    "verification_ms") if any(
                        row["parameter_check_performed"] for row in rows) else 0.0,
                "parameter_checks": sum(
                    bool(row["parameter_check_performed"]) for row in rows),
                "final_loss": losses[-1],
            })
    policies = {}
    for policy, interval in POLICIES:
        rows = [row for row in process_summaries if row["policy"] == policy]
        policies[policy] = {
            "parameter_check_interval": interval,
            "processes": len(rows),
            "median_total_ms": median(rows, "median_total_ms"),
            "median_verification_ms": median(rows, "median_verification_ms"),
            "parameter_checks_per_process": int(median(rows, "parameter_checks")),
            "final_loss": median(rows, "final_loss"),
        }
    baseline = policies["every_step"]["median_total_ms"]
    for policy in ("final_step", "disabled"):
        policies[policy]["speedup_vs_every_step"] = (
            baseline / policies[policy]["median_total_ms"])
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_verification_matrix_summary",
        "raw_records": len(records),
        "processes": len(process_summaries),
        "runs_per_policy": args.runs,
        "steps_per_process": args.steps,
        "steady_steps_per_process": args.steps - 1,
        "aggregation": "median of per-process step-2..20 medians with rotated policy order",
        "policies": policies,
        "loss_trajectories_exact": True,
        "decision": "retain default interval 1; expose sparse/disabled production measurements",
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
        print(f"data_parallel_verification_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)

