#!/usr/bin/env python3
"""Repeat official T512 BF16 grouped gate/up capability probes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


MODELS = ("qwen", "deepseek")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--maximum-algorithms", type=int, default=64)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0 or
            result.maximum_algorithms <= 0 or result.maximum_algorithms > 256 or
            not result.binary.is_file()):
        parser.error("binary and positive grouped gate/up options are required")
    return result


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("grouped gate/up probe emitted no JSON")


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        order = list(MODELS)
        if process_run % 2 == 0:
            order.reverse()
        for model in order:
            completed = subprocess.run([
                str(args.binary), "--model", model,
                "--projection", "gate-up", "--rows", "512",
                "--output-dtype", "bf16", "--equal-width", "false",
                "--maximum-algorithms", str(args.maximum_algorithms),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ], text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            record = last_json(completed.stdout)
            if record.get("status") != "pass" or \
                    record.get("record_type") != \
                        "bf16_grouped_gate_up_probe" or \
                    record.get("projection") != "gate-up" or \
                    record.get("model") != model or \
                    record.get("groups") != 2 or \
                    record.get("grouped_supported") is not True or \
                    int(record.get("passing_candidates", 0)) != \
                        args.maximum_algorithms:
                raise RuntimeError("invalid grouped gate/up probe record")
            record.update({
                "process_run": process_run,
                "process_order": order,
            })
            records.append(record)

    comparisons = []
    for model in MODELS:
        rows = [row for row in records if row["model"] == model]
        comparisons.append({
            "model": model,
            "rows": 512,
            "groups": 2,
            "solution_indices": sorted({
                int(row["solution_index"]) for row in rows}),
            "algorithm_count": min(
                int(row["algorithm_count"]) for row in rows),
            "passing_candidates": min(
                int(row["passing_candidates"]) for row in rows),
            "maximum_absolute_error": max(
                float(row["maximum_absolute_error"]) for row in rows),
            "maximum_rms_error": max(
                float(row["maximum_rms_error"]) for row in rows),
            "event_speedup_median": statistics.median(
                float(row["event_speedup"]) for row in rows),
            "wall_speedup_median": statistics.median(
                float(row["wall_speedup"]) for row in rows),
            "reinitialized_event_speedup_median": statistics.median(
                float(row["reinitialized_event_speedup"]) for row in rows),
            "reinitialized_wall_speedup_median": statistics.median(
                float(row["reinitialized_wall_speedup"]) for row in rows),
            "user_arguments_setup_ms_median": statistics.median(
                float(row["user_arguments_setup_ms"]) for row in rows),
            "user_arguments_event_speedup_median": statistics.median(
                float(row["user_arguments_event_speedup"]) for row in rows),
            "user_arguments_wall_speedup_median": statistics.median(
                float(row["user_arguments_wall_speedup"]) for row in rows),
            "workspace_bytes": max(int(row["workspace_bytes"]) for row in rows),
        })
    capability = all(
        row["algorithm_count"] > 0 and row["passing_candidates"] ==
            args.maximum_algorithms and row["maximum_absolute_error"] == 0 and
        row["user_arguments_event_speedup_median"] >= 1.05
        for row in comparisons)
    reinitialization_counterexample = all(
        row["reinitialized_event_speedup_median"] < 1.0
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "bf16_grouped_gate_up_matrix_summary",
        "raw_processes": len(records),
        "capability_gate": capability,
        "reinitialization_counterexample_gate":
            reinitialization_counterexample,
        "comparisons": comparisons,
        "decision": (
            "continue to pointer-stable FFN Arena model gate"
            if capability else "reject grouped gate/up capability"),
    }
    with (args.output_directory / "raw.jsonl").open(
            "w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if capability else 2


if __name__ == "__main__":
    raise SystemExit(main())
