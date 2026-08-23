#!/usr/bin/env python3
"""Repeat the BF16 training-shape solution tuner in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys


SHAPES = (
    ("qwen_qo", 512, 896, 896),
    ("qwen_kv", 512, 896, 128),
    ("qwen_gate_up", 512, 896, 4864),
    ("qwen_down", 512, 4864, 896),
    ("deep_qo", 512, 1536, 1536),
    ("deep_kv", 512, 1536, 256),
    ("deep_gate_up", 512, 1536, 8960),
    ("deep_down", 512, 8960, 1536),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--maximum-algorithms", type=int, default=64)
    parser.add_argument("--workspace-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    result = parser.parse_args()
    if result.runs <= 0 or result.maximum_algorithms <= 0 or \
            result.workspace_bytes < 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("invalid run/timing/algorithm options")
    return result


def run(args: argparse.Namespace, shape: tuple[str, int, int, int],
        process_run: int) -> dict:
    name, rows, inner, columns = shape
    command = [
        str(args.binary), "--rows", str(rows), "--inner", str(inner),
        "--columns", str(columns), "--output-dtype", "fp32",
        "--maximum-algorithms", str(args.maximum_algorithms),
        "--workspace-bytes", str(args.workspace_bytes),
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
    ]
    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    completed = subprocess.run(
        command, env=environment, text=True, capture_output=True,
        timeout=args.timeout, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{name}/run{process_run}: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    result["shape_name"] = name
    result["process_run"] = process_run
    return result


def summarize(shape: tuple[str, int, int, int], records: list[dict]) -> dict:
    name, rows, inner, columns = shape
    candidates_by_run = [
        {candidate["index"]: candidate for candidate in record["candidates"]
         if candidate["supported"] and candidate["correctness_passed"]}
        for record in records
    ]
    common = set(candidates_by_run[0])
    for current in candidates_by_run[1:]:
        common &= set(current)
    if not common:
        raise ValueError(f"{name} has no common passing solution")
    medians = {
        index: statistics.median(current[index]["event_ms_p50"]
                                 for current in candidates_by_run)
        for index in common
    }
    recommended = min(medians, key=lambda index: (medians[index], index))
    default = statistics.median(record["default_event_ms_p50"] for record in records)
    maximum_error = max(
        candidate["maximum_absolute_error"]
        for current in candidates_by_run for candidate in current.values())
    maximum_rms = max(
        candidate["rms_error"]
        for current in candidates_by_run for candidate in current.values())
    return {
        "schema_version": 1,
        "status": "pass",
        "shape_name": name,
        "rows": rows,
        "inner": inner,
        "columns": columns,
        "runs": len(records),
        "common_passing_candidates": len(common),
        "default_event_ms_p50_median": default,
        "recommended_index": recommended,
        "recommended_event_ms_p50_median": medians[recommended],
        "operator_speedup": default / medians[recommended],
        "maximum_absolute_error": maximum_error,
        "maximum_rms_error": maximum_rms,
        "per_run_recommendations": [record["recommended_index"] for record in records],
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw = []
    summaries = []
    for shape in SHAPES:
        records = [run(args, shape, process_run + 1)
                   for process_run in range(args.runs)]
        raw.extend(records)
        summary = summarize(shape, records)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "track": "bf16_training_solution_matrix",
        "runs_per_shape": args.runs,
        "workspace_limit_bytes": args.workspace_bytes,
        "raw_rows": len(raw),
        "summaries": summaries,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"bf16_training_solution_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
