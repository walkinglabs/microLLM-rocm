#!/usr/bin/env python3
"""Measure cast-inclusive BF16 weight gradients on pinned model shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


SHAPES = (
    ("qwen2.5-0.5b", "query", 896, 896),
    ("qwen2.5-0.5b", "kv", 896, 128),
    ("qwen2.5-0.5b", "gate", 896, 4864),
    ("deepseek-r1-distill-qwen-1.5b", "query", 1536, 1536),
    ("deepseek-r1-distill-qwen-1.5b", "kv", 1536, 256),
    ("deepseek-r1-distill-qwen-1.5b", "gate", 1536, 8960),
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
    if not args.binary.is_file():
        parser.error("BF16 weight-gradient benchmark is unavailable")
    if args.runs <= 0 or args.rows <= 0 or args.warmup < 0 or args.repetitions <= 0:
        parser.error("matrix counts must be positive and warmup nonnegative")
    return args


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("BF16 weight-gradient worker emitted no JSON")


def run(args: argparse.Namespace, model: str, family: str,
        hidden: int, width: int, process_run: int) -> dict:
    command = [
        str(args.binary), "--rows", str(args.rows),
        "--hidden", str(hidden), "--width", str(width),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    record = last_json(completed.stdout)
    required = (
        record.get("status") == "pass" and
        record.get("record_type") == "bf16_weight_gradient_probe" and
        record.get("rows") == args.rows and record.get("hidden") == hidden and
        record.get("width") == width and
        record.get("candidate_includes_input_cast_transpose") is True and
        record.get("candidate_includes_gradient_cast") is True and
        record.get("candidate_output_dtype") == "float32" and
        record.get("complete_output_finite") is True and
        record.get("complete_output_elements") == hidden * width and
        record.get("bf16_reference_samples") == 64 and
        float(record.get("bf16_reference_sample_max_error", math.inf)) <= 2.0e-3 and
        math.isfinite(float(record.get("fp32_baseline_max_error", math.inf))) and
        math.isfinite(float(record.get("fp32_baseline_rms_error", math.inf))) and
        float(record.get("event_speedup", 0.0)) > 0.0
    )
    if not required:
        raise RuntimeError(f"BF16 weight-gradient gate failed for {model}/{family}")
    record.update({
        "model": model,
        "family": family,
        "process_run": process_run,
        "record_type": "bf16_weight_gradient_matrix_measurement",
    })
    return record


def main() -> int:
    args = options()
    records = []
    summaries = []
    for model, family, hidden, width in SHAPES:
        rows = [run(args, model, family, hidden, width, process_run)
                for process_run in range(1, args.runs + 1)]
        records.extend(rows)
        speedups = [float(row["event_speedup"]) for row in rows]
        summary = {
            "model": model,
            "family": family,
            "rows": args.rows,
            "hidden": hidden,
            "width": width,
            "processes": len(rows),
            "event_speedup_median": statistics.median(speedups),
            "event_speedup_minimum": min(speedups),
            "event_speedup_maximum": max(speedups),
            "fp32_baseline_max_error_maximum": max(
                float(row["fp32_baseline_max_error"]) for row in rows),
            "fp32_baseline_rms_error_maximum": max(
                float(row["fp32_baseline_rms_error"]) for row in rows),
            "bf16_reference_sample_max_error_maximum": max(
                float(row["bf16_reference_sample_max_error"]) for row in rows),
        }
        summary["passes_operator_performance_gate"] = (
            summary["event_speedup_median"] >= 1.05 and
            summary["event_speedup_minimum"] >= 1.0)
        summaries.append(summary)
    eligible = [row for row in summaries
                if row["passes_operator_performance_gate"]]
    model_large_gate = all(any(
        row["model"] == model and row["family"] in {"query", "gate"} and
        row["passes_operator_performance_gate"]
        for row in summaries) for model in {
            "qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"})
    result = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "bf16_weight_gradient_matrix_summary",
        "raw_processes": len(records),
        "rows": args.rows,
        "runs_per_shape": args.runs,
        "shape_count": len(summaries),
        "performance_gate": "median >= 1.05 and minimum >= 1.0",
        "eligible_shape_count": len(eligible),
        "both_models_have_large_family_candidate": model_large_gate,
        "shapes": summaries,
        "decision": ("admit only eligible shapes to an explicit Autograd model gate"
                     if model_large_gate else
                     "reject model integration; retain the benchmark counterexample"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"bf16_weight_gradient_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
