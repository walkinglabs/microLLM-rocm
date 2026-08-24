#!/usr/bin/env python3
"""Measure caller-owned hipBLASLt GEMM replay on official training shapes."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


DEFAULT_SHAPES = (
    "qwen:512:896:896",
    "deepseek:512:1536:1536",
)


def parse_positive_csv(text: str, name: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return values


def parse_shape(text: str) -> dict:
    fields = text.split(":")
    if len(fields) != 4 or not fields[0]:
        raise argparse.ArgumentTypeError("shape must be name:rows:inner:columns")
    try:
        rows, inner, columns = (int(value) for value in fields[1:])
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    if min(rows, inner, columns) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return {"name": fields[0], "rows": rows, "inner": inner, "columns": columns}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--calls", default="1,8,32")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    result = parser.parse_args()
    result.shapes = [parse_shape(value) for value in result.shapes.split(",")]
    result.calls = parse_positive_csv(result.calls, "--calls")
    if result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs/repetitions must be positive and warmup nonnegative")
    if not result.binary.is_file():
        parser.error(f"benchmark binary does not exist: {result.binary}")
    return result


def command(args: argparse.Namespace, mode: str, calls: int,
            shape: dict) -> list[str]:
    return [
        str(args.binary),
        "--mode", mode,
        "--calls", str(calls),
        "--rows", str(shape["rows"]),
        "--inner", str(shape["inner"]),
        "--columns", str(shape["columns"]),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ]


def execute(args: argparse.Namespace, mode: str, calls: int,
            shape: dict) -> dict:
    completed = subprocess.run(
        command(args, mode, calls, shape), check=True, text=True,
        capture_output=True)
    record = json.loads(completed.stdout)
    if record.get("status") != "pass" or record.get("mode") != mode:
        raise RuntimeError("HIP Graph GEMM benchmark returned an invalid status/mode")
    if any(record.get(field) != shape[field] for field in ("rows", "inner", "columns")):
        raise RuntimeError("HIP Graph GEMM benchmark returned the wrong shape")
    record["shape_name"] = shape["name"]
    return record


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def summarize(records: list[dict], shapes: list[dict],
              calls_values: list[int], runs: int) -> dict:
    comparisons = []
    for shape in shapes:
        for calls in calls_values:
            selected = [
                row for row in records
                if row["shape_name"] == shape["name"] and row["calls"] == calls
            ]
            policies = {}
            for mode in ("eager", "graph"):
                group = [row for row in selected if row["mode"] == mode]
                if len(group) != runs:
                    raise RuntimeError(f"{shape['name']}/{calls}/{mode} is incomplete")
                policies[mode] = {
                    "event_median_ms": median(group, "event_median_ms"),
                    "wall_median_ms": median(group, "wall_median_ms"),
                    "wall_p95_ms": median(group, "wall_p95_ms"),
                    "setup_ms": median(group, "setup_ms"),
                    "captured_nodes": median(group, "captured_nodes"),
                }
            comparisons.append({
                "shape_name": shape["name"],
                "rows": shape["rows"],
                "inner": shape["inner"],
                "columns": shape["columns"],
                "calls": calls,
                "policies": policies,
                "event_speedup": (
                    policies["eager"]["event_median_ms"] /
                    policies["graph"]["event_median_ms"]),
                "wall_speedup": (
                    policies["eager"]["wall_median_ms"] /
                    policies["graph"]["wall_median_ms"]),
                "captured_node_count_correct": (
                    policies["graph"]["captured_nodes"] == calls),
            })
    exact = all(
        float(row.get("maximum_absolute_error", math.inf)) == 0.0 and
        float(row.get("rms_error", math.inf)) == 0.0 and
        row.get("output_address_stable") is True and
        row.get("host_to_device_calls") == 0 and
        row.get("device_to_host_calls") == 0 and
        row.get("device_to_device_calls") == 0
        for row in records)
    repeated = [row for row in comparisons if row["calls"] >= 8]
    gates = {
        "exact_stable_transfer_free": exact,
        "captured_node_count": all(
            row["captured_node_count_correct"] for row in comparisons),
        "wall_speedup_calls_ge_8": all(
            row["wall_speedup"] >= 1.02 for row in repeated),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "aggregation": "median of fresh processes with alternating mode order",
        "runs_per_case_mode": runs,
        "comparisons": comparisons,
        "keep_gate": {
            "maximum_absolute_and_rms_error": 0.0,
            "stable_output_address": True,
            "timed_payload_transfers": 0,
            "captured_nodes": "one node per hipBLASLt call",
            "wall_speedup_minimum_for_calls_ge_8": 1.02,
        },
        "gate_results": gates,
        "decision": (
            "keep caller-owned hipBLASLt Graph boundary"
            if all(gates.values()) else
            "reject caller-owned hipBLASLt Graph boundary"),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        for shape in args.shapes:
            for calls in args.calls:
                modes = ("eager", "graph") if process_run % 2 else ("graph", "eager")
                for mode in modes:
                    record = execute(args, mode, calls, shape)
                    record["process_run"] = process_run
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize(records, args.shapes, args.calls, args.runs)
    (args.output_directory / "matrix.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            subprocess.CalledProcessError, RuntimeError) as error:
        print(f"hip_graph_gemm_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
