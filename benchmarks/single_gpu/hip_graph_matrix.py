#!/usr/bin/env python3
"""Measure caller-owned HIP Graph replay against eager launches."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


def positive_csv(text: str, name: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--nodes", default="1,8,32,128,512")
    parser.add_argument("--elements", default="1,4096")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    result = parser.parse_args()
    result.nodes = positive_csv(result.nodes, "--nodes")
    result.elements = positive_csv(result.elements, "--elements")
    if result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs/repetitions must be positive and warmup nonnegative")
    if not result.binary.is_file():
        parser.error(f"benchmark binary does not exist: {result.binary}")
    return result


def command(args: argparse.Namespace, mode: str, nodes: int,
            elements: int) -> list[str]:
    return [
        str(args.binary),
        "--mode", mode,
        "--nodes", str(nodes),
        "--elements", str(elements),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ]


def execute(args: argparse.Namespace, mode: str, nodes: int,
            elements: int) -> dict:
    completed = subprocess.run(
        command(args, mode, nodes, elements), check=True, text=True,
        capture_output=True)
    record = json.loads(completed.stdout)
    if record.get("status") != "pass" or record.get("mode") != mode:
        raise RuntimeError("HIP Graph benchmark returned an invalid status/mode")
    if record.get("nodes") != nodes or record.get("elements") != elements:
        raise RuntimeError("HIP Graph benchmark returned the wrong shape")
    if float(record.get("maximum_absolute_error", math.inf)) != 0.0:
        raise RuntimeError("HIP Graph benchmark output is not exact")
    return record


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def summarize(records: list[dict], nodes_values: list[int],
              element_values: list[int], runs: int) -> dict:
    comparisons = []
    for elements in element_values:
        for nodes in nodes_values:
            selected = [
                row for row in records
                if row["nodes"] == nodes and row["elements"] == elements
            ]
            policies = {}
            for mode in ("eager", "graph"):
                group = [row for row in selected if row["mode"] == mode]
                if len(group) != runs:
                    raise RuntimeError(
                        f"nodes={nodes}/elements={elements}/{mode} is incomplete")
                policies[mode] = {
                    "event_median_ms": median(group, "event_median_ms"),
                    "wall_median_ms": median(group, "wall_median_ms"),
                    "wall_p95_ms": median(group, "wall_p95_ms"),
                    "setup_ms": median(group, "setup_ms"),
                    "captured_nodes": median(group, "captured_nodes"),
                }
            comparisons.append({
                "nodes": nodes,
                "elements": elements,
                "policies": policies,
                "event_speedup": (
                    policies["eager"]["event_median_ms"] /
                    policies["graph"]["event_median_ms"]),
                "wall_speedup": (
                    policies["eager"]["wall_median_ms"] /
                    policies["graph"]["wall_median_ms"]),
                "captured_node_count_correct": (
                    policies["graph"]["captured_nodes"] == nodes + 1),
            })

    exact_and_transfer_free = all(
        float(row["maximum_absolute_error"]) == 0.0 and
        row["host_to_device_calls"] == 0 and
        row["device_to_host_calls"] == 0 and
        row["device_to_device_calls"] == 0
        for row in records)
    large_node_rows = [row for row in comparisons if row["nodes"] >= 32]
    gate_results = {
        "exact_and_transfer_free": exact_and_transfer_free,
        "captured_node_count": all(
            row["captured_node_count_correct"] for row in comparisons),
        "wall_speedup_nodes_ge_32": all(
            row["wall_speedup"] >= 1.05 for row in large_node_rows),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "aggregation": "median of fresh processes with alternating mode order",
        "runs_per_case_mode": runs,
        "comparisons": comparisons,
        "keep_gate": {
            "maximum_absolute_error": 0.0,
            "timed_payload_transfers": 0,
            "captured_nodes": "requested add nodes plus one fill node",
            "wall_speedup_minimum_for_nodes_ge_32": 1.05,
        },
        "gate_results": gate_results,
        "decision": (
            "keep caller-owned HIP Graph runtime primitive"
            if all(gate_results.values()) else
            "reject caller-owned HIP Graph runtime primitive"),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        for elements in args.elements:
            for nodes in args.nodes:
                modes = ("eager", "graph") if process_run % 2 else ("graph", "eager")
                for mode in modes:
                    record = execute(args, mode, nodes, elements)
                    record["process_run"] = process_run
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize(records, args.nodes, args.elements, args.runs)
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
        print(f"hip_graph_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
