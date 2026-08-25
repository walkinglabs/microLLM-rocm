#!/usr/bin/env python3
"""Compare allocating and preallocated BF16 weight-gradient execution."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


SHAPES = (
    ("qwen2.5-0.5b", 896, 4864),
    ("deepseek-r1-distill-qwen-1.5b", 1536, 8960),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.rows <= 0 or \
            args.warmup < 0 or args.repetitions <= 0:
        parser.error("workspace matrix inputs are invalid")
    return args


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("workspace worker emitted no JSON")


def main() -> int:
    args = options()
    records = []
    summaries = []
    for model, hidden, width in SHAPES:
        rows = []
        for process_run in range(1, args.runs + 1):
            completed = subprocess.run([
                str(args.binary), "--rows", str(args.rows),
                "--hidden", str(hidden), "--width", str(width),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ], text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            record = last_json(completed.stdout)
            if (record.get("status") != "pass" or
                    record.get("complete_output_finite") is not True or
                    record.get("allocating_allocation_calls_per_invocation") != 3.0 or
                    record.get("allocating_backend_allocation_calls_per_invocation") != 0.0 or
                    record.get("allocating_cache_reuse_calls_per_invocation") != 3.0 or
                    float(record.get("preallocated_over_allocating_wall_speedup", 0.0)) <= 0.0 or
                    float(record.get("preallocated_over_allocating_event_speedup", 0.0)) <= 0.0):
                raise RuntimeError(f"workspace cost gate failed for {model}")
            record.update({
                "record_type": "bf16_weight_gradient_workspace_measurement",
                "model": model, "process_run": process_run,
            })
            rows.append(record)
            records.append(record)
        wall = [float(row["preallocated_over_allocating_wall_speedup"])
                for row in rows]
        event = [float(row["preallocated_over_allocating_event_speedup"])
                 for row in rows]
        item = {
            "model": model,
            "rows": args.rows,
            "hidden": hidden,
            "width": width,
            "processes": len(rows),
            "wall_speedup_median": statistics.median(wall),
            "wall_speedup_minimum": min(wall),
            "wall_speedup_maximum": max(wall),
            "event_speedup_median": statistics.median(event),
            "event_speedup_minimum": min(event),
            "event_speedup_maximum": max(event),
        }
        item["passes_workspace_gate"] = (
            item["wall_speedup_median"] >= 1.01 and
            item["wall_speedup_minimum"] >= 1.0)
        summaries.append(item)
    accepted = all(row["passes_workspace_gate"] for row in summaries)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "bf16_weight_gradient_workspace_summary",
        "raw_processes": len(records),
        "runs_per_shape": args.runs,
        "shape_count": len(summaries),
        "gate": "wall median >= 1.01 and minimum >= 1.0 for both models",
        "models": summaries,
        "decision": ("design a caller-owned workspace API"
                     if accepted else
                     "reject workspace API; cached logical allocation cost is too small"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"bf16_weight_gradient_workspace_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)

