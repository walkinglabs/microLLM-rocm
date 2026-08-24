#!/usr/bin/env python3
"""Screen grouped QKV and gate/up at flattened rows 256 and 1024."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


MODELS = ("qwen", "deepseek")
ROWS = (256, 1024)
PROJECTIONS = (("qkv", "model"), ("gate-up", "bf16"))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--maximum-algorithms", type=int, default=64)
    parser.add_argument("--minimum-user-arguments-speedup",
                        type=float, default=1.05)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0 or
            result.maximum_algorithms <= 0 or result.maximum_algorithms > 256 or
            result.minimum_user_arguments_speedup <= 1 or
            not result.binary.is_file()):
        parser.error("grouped shape matrix options are invalid")
    return result


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("grouped shape probe emitted no JSON")


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        cases = [
            (rows, model, projection, mode)
            for rows in ROWS for model in MODELS
            for projection, mode in PROJECTIONS
        ]
        if process_run % 2 == 0:
            cases.reverse()
        for rows, model, projection, mode in cases:
            completed = subprocess.run([
                str(args.binary), "--model", model,
                "--projection", projection, "--rows", str(rows),
                "--output-dtype", mode, "--equal-width", "false",
                "--maximum-algorithms", str(args.maximum_algorithms),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ], text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            record = last_json(completed.stdout)
            expected_type = (
                "bf16_grouped_qkv_probe"
                if projection == "qkv" else
                "bf16_grouped_gate_up_probe")
            expected_groups = 3 if projection == "qkv" else 2
            if record.get("status") != "pass" or \
                    record.get("record_type") != expected_type or \
                    record.get("projection") != projection or \
                    record.get("model") != model or \
                    record.get("rows") != rows or \
                    record.get("groups") != expected_groups or \
                    record.get("grouped_supported") is not True or \
                    int(record.get("passing_candidates", 0)) != \
                        args.maximum_algorithms:
                raise RuntimeError("invalid grouped shape probe record")
            record.update({
                "process_run": process_run,
                "process_order": [
                    f"{item[0]}:{item[1]}:{item[2]}"
                    for item in cases],
            })
            records.append(record)

    comparisons = []
    for rows in ROWS:
        for model in MODELS:
            for projection, _ in PROJECTIONS:
                selected = [
                    row for row in records
                    if row["rows"] == rows and row["model"] == model and
                    row["projection"] == projection]
                comparisons.append({
                    "rows": rows,
                    "model": model,
                    "projection": projection,
                    "groups": 3 if projection == "qkv" else 2,
                    "algorithm_count": min(
                        int(row["algorithm_count"]) for row in selected),
                    "passing_candidates": min(
                        int(row["passing_candidates"]) for row in selected),
                    "solution_indices": sorted({
                        int(row["solution_index"]) for row in selected}),
                    "maximum_absolute_error": max(
                        float(row["maximum_absolute_error"])
                        for row in selected),
                    "maximum_rms_error": max(
                        float(row["maximum_rms_error"])
                        for row in selected),
                    "event_speedup_median": statistics.median(
                        float(row["event_speedup"]) for row in selected),
                    "user_arguments_event_speedup_median": statistics.median(
                        float(row["user_arguments_event_speedup"])
                        for row in selected),
                    "user_arguments_wall_speedup_median": statistics.median(
                        float(row["user_arguments_wall_speedup"])
                        for row in selected),
                    "reinitialized_event_speedup_median": statistics.median(
                        float(row["reinitialized_event_speedup"])
                        for row in selected),
                    "user_arguments_setup_ms_median": statistics.median(
                        float(row["user_arguments_setup_ms"])
                        for row in selected),
                })
    capability = all(
        row["algorithm_count"] > 0 and
        row["passing_candidates"] == args.maximum_algorithms and
        row["user_arguments_event_speedup_median"] >=
            args.minimum_user_arguments_speedup
        for row in comparisons)
    reinitialization_faster_cases = sum(
        row["reinitialized_event_speedup_median"] > 1.0
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if capability else "fail",
        "record_type": "bf16_grouped_shape_matrix_summary",
        "raw_processes": len(records),
        "comparisons": comparisons,
        "capability_gate": capability,
        "reinitialization_faster_cases":
            reinitialization_faster_cases,
        "decision": (
            "continue to cross-shape complete-model gate"
            if capability else "reject grouped cross-shape capability"),
    }
    with (args.output_directory / "raw.jsonl").open(
            "w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
