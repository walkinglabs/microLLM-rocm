#!/usr/bin/env python3
"""Rotated-process MI300 gate for caller-owned weight-gradient output."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


SHAPES = (
    ("model_s_head_t32", 32, 384, 8192, True),
    ("model_s_ffn_t32", 32, 384, 832, True),
    ("model_s_attention_t32", 32, 384, 384, True),
    ("model_s_head_t512", 512, 384, 8192, True),
    ("tiny_counterexample", 32, 64, 64, False),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=40)
    args = parser.parse_args()
    if (not args.binary.is_file() or args.runs <= 0 or args.warmup < 0 or
            args.repetitions <= 0):
        parser.error("gradient producer matrix inputs are invalid")
    return args


def execute(args: argparse.Namespace, name: str, rows: int, hidden: int,
            width: int, process_run: int) -> dict:
    order = "allocating-first" if process_run % 2 else "direct-first"
    completed = subprocess.run([
        str(args.binary), "--rows", str(rows), "--hidden", str(hidden),
        "--width", str(width), "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions), "--order", order,
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    lines = [line for line in completed.stdout.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"{name} emitted an unexpected record count")
    row = json.loads(lines[0])
    expected_elements = hidden * width
    if (row.get("status") != "pass" or
            row.get("record_type") != "gradient_producer_out_probe" or
            row.get("rows") != rows or row.get("hidden") != hidden or
            row.get("width") != width or row.get("order") != order or
            row.get("complete_output_elements") != expected_elements or
            row.get("complete_output_max_error") != 0.0 or
            row.get("complete_output_rms_error") != 0.0 or
            row.get("allocating_calls_per_invocation") != 1.0 or
            row.get("direct_calls_per_invocation") != 0.0):
        raise RuntimeError(f"{name} producer contract changed")
    row.update({
        "shape": name,
        "model_s_shape": next(item[4] for item in SHAPES if item[0] == name),
        "process_run": process_run,
    })
    return row


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    raw = []
    for process_run in range(1, args.runs + 1):
        ordered = SHAPES if process_run % 2 else tuple(reversed(SHAPES))
        for name, rows, hidden, width, _ in ordered:
            raw.append(execute(args, name, rows, hidden, width, process_run))
    shapes = {}
    for name, rows, hidden, width, model_s_shape in SHAPES:
        records = [row for row in raw if row["shape"] == name]
        allocating_event = median(records, "allocating_event_ms_p50")
        direct_event = median(records, "direct_event_ms_p50")
        allocating_wall = median(records, "allocating_wall_ms_p50")
        direct_wall = median(records, "direct_wall_ms_p50")
        event_speedup = allocating_event / direct_event
        wall_speedup = allocating_wall / direct_wall
        shapes[name] = {
            "rows": rows,
            "hidden": hidden,
            "width": width,
            "model_s_shape": model_s_shape,
            "processes": len(records),
            "allocating_event_ms_p50": allocating_event,
            "direct_event_ms_p50": direct_event,
            "allocating_wall_ms_p50": allocating_wall,
            "direct_wall_ms_p50": direct_wall,
            "event_speedup": event_speedup,
            "wall_speedup": wall_speedup,
            "passes_1_05_gate": event_speedup >= 1.05 and wall_speedup >= 1.05,
        }
    admitted = [name for name, row in shapes.items()
                if row["model_s_shape"] and row["passes_1_05_gate"]]
    rejected = [name for name, row in shapes.items()
                if not row["passes_1_05_gate"]]
    decision = ("admit exact producer shapes to Autograd gate"
                if admitted else
                "close direct producer route before Autograd")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "gradient_producer_out_matrix_summary",
        "raw_records": len(raw),
        "runs_per_shape": args.runs,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "aggregation": "median of fresh-process p50 with alternating operation and shape order",
        "complete_outputs_exact": True,
        "allocating_calls_per_invocation": 1.0,
        "direct_calls_per_invocation": 0.0,
        "shapes": shapes,
        "admitted_model_s_shapes": admitted,
        "rejected_shapes": rejected,
        "decision": decision,
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
        print(f"gradient_producer_out_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
