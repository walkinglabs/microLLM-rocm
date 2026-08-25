#!/usr/bin/env python3
"""Model-S two-rank gradient-ready order and natural-bucket audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BUCKET_BYTES = 25 * 1024 * 1024


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.steps <= 0:
        parser.error("gradient-ready audit inputs are invalid")
    return args


def execute(args: argparse.Namespace, process_run: int) -> list[dict]:
    completed = subprocess.run([
        str(args.binary), "--model", "model-s", "--steps", str(args.steps),
        "--context", "32", "--batch", "1",
        "--bucket-bytes", str(BUCKET_BYTES),
        "--parameter-check-interval", str(args.steps),
        "--record-gradient-ready-order", "true", "--seed", "601",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != args.steps:
        raise RuntimeError("gradient-ready CLI emitted an unexpected step count")
    for row in rows:
        names = row.get("parameter_names", [])
        elements = row.get("parameter_elements", [])
        rank0 = row.get("gradient_ready_order_rank0", [])
        rank1 = row.get("gradient_ready_order_rank1", [])
        if (row.get("record_gradient_ready_order") is not True or
                row.get("gradient_ready_audit_performed") is not True or
                row.get("gradient_ready_orders_match") is not True or
                len(names) != 57 or len(elements) != 57 or
                sorted(rank0) != list(range(57)) or rank1 != rank0):
            raise RuntimeError("gradient-ready step contract changed")
        row.update({
            "record_type": "data_parallel_gradient_ready_measurement",
            "process_run": process_run,
        })
    if (rows[-1].get("parameter_check_performed") is not True or
            rows[-1].get("parameter_max_difference") != 0.0):
        raise RuntimeError("gradient-ready final parameter gate failed")
    return rows


def bucket_ranges(elements: list[int]) -> list[tuple[int, int, int]]:
    maximum_elements = BUCKET_BYTES // 4
    ranges = []
    first = 0
    while first < len(elements):
        end = first
        total = 0
        while end < len(elements):
            count = int(elements[end])
            if end != first and total + count > maximum_elements:
                break
            total += count
            end += 1
            if total >= maximum_elements:
                break
        ranges.append((first, end, total))
        first = end
    return ranges


def main() -> int:
    args = options()
    raw = []
    reference_order = None
    names = None
    elements = None
    for process_run in range(1, args.runs + 1):
        rows = execute(args, process_run)
        for row in rows:
            order = row["gradient_ready_order_rank0"]
            if reference_order is None:
                reference_order = order
                names = row["parameter_names"]
                elements = row["parameter_elements"]
            elif (order != reference_order or row["parameter_names"] != names or
                  row["parameter_elements"] != elements):
                raise RuntimeError("gradient-ready order changed across steps/processes")
        raw.extend(rows)
    assert reference_order is not None and names is not None and elements is not None
    positions = {parameter: position
                 for position, parameter in enumerate(reference_order)}
    bucket_rows = []
    for bucket, (first, end, total) in enumerate(bucket_ranges(elements)):
        ready_positions = [positions[index] for index in range(first, end)]
        completion_position = max(ready_positions)
        bucket_rows.append({
            "bucket": bucket,
            "first_parameter": first,
            "end_parameter": end,
            "parameter_count": end - first,
            "elements": total,
            "bytes": total * 4,
            "first_name": names[first],
            "last_name": names[end - 1],
            "first_ready_position": min(ready_positions),
            "completion_position": completion_position,
            "completion_fraction": (completion_position + 1) / len(reference_order),
            "ready_before_backward_end": completion_position < len(reference_order) - 1,
        })
    early = sum(1 for row in bucket_rows if row["ready_before_backward_end"])
    reverse_order = list(reversed(range(len(reference_order))))
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "data_parallel_gradient_ready_summary",
        "raw_records": len(raw),
        "processes": args.runs,
        "steps_per_process": args.steps,
        "parameter_count": len(reference_order),
        "bucket_bytes": BUCKET_BYTES,
        "bucket_count": len(bucket_rows),
        "orders_match_across_ranks_steps_processes": True,
        "ready_order_is_reverse_parameter_order": reference_order == reverse_order,
        "ready_order": reference_order,
        "parameter_names": names,
        "parameter_elements": elements,
        "buckets": bucket_rows,
        "buckets_ready_before_backward_end": early,
        "decision": ("admit event-based overlap prototype"
                     if early >= 2 else
                     "reorder buckets or close overlap hypothesis"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
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
        print(f"data_parallel_gradient_ready_audit: {error}", file=sys.stderr)
        raise SystemExit(2)
