#!/usr/bin/env python3
"""Measure stable two-slot activation arena eager and Graph policies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


def values(text: str) -> list[int]:
    result = [int(item) for item in text.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--nodes", type=values, default=[8, 32, 128, 512])
    parser.add_argument("--elements", type=values, default=[1, 4096])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed-repetitions", type=int, default=20)
    result = parser.parse_args()
    if result.repetitions <= 0 or result.warmup < 0 or result.timed_repetitions <= 0:
        parser.error("repetitions/timed-repetitions must be positive; warmup nonnegative")
    return result


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            return row
    raise RuntimeError("benchmark emitted no JSON record")


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    modes = ["deferred", "arena", "arena_graph"]
    raw: list[dict] = []
    for nodes in args.nodes:
        for elements in args.elements:
            for repetition in range(1, args.repetitions + 1):
                offset = (repetition - 1) % len(modes)
                order = modes[offset:] + modes[:offset]
                for mode in order:
                    completed = subprocess.run([
                        str(args.binary), "--mode", mode,
                        "--nodes", str(nodes), "--elements", str(elements),
                        "--warmup", str(args.warmup),
                        "--repetitions", str(args.timed_repetitions)],
                        text=True, capture_output=True, check=False)
                    stem = f"n{nodes}-e{elements}-r{repetition}-{mode}"
                    (logs / f"{stem}.stdout.txt").write_text(
                        completed.stdout, encoding="utf-8")
                    (logs / f"{stem}.stderr.txt").write_text(
                        completed.stderr, encoding="utf-8")
                    if completed.returncode != 0:
                        raise RuntimeError(f"benchmark failed for {stem}: {completed.stderr}")
                    row = last_json(completed.stdout)
                    row["process_run"] = repetition
                    row["process_order"] = order
                    if row.get("status") != "pass" or row.get("mode") != mode:
                        raise RuntimeError(f"invalid benchmark row for {stem}")
                    raw.append(row)
    comparisons: list[dict] = []
    for nodes in args.nodes:
        for elements in args.elements:
            selected = [row for row in raw
                        if row["nodes"] == nodes and row["elements"] == elements]
            grouped = {mode: [row for row in selected if row["mode"] == mode]
                       for mode in modes}
            median = lambda mode, field: statistics.median(
                float(row[field]) for row in grouped[mode])
            deferred = median("deferred", "wall_p50_ms")
            arena = median("arena", "wall_p50_ms")
            graph = median("arena_graph", "wall_p50_ms")
            expected_capacity = ((elements * 4 + 255) // 256) * 256 * 2
            comparisons.append({
                "nodes": nodes, "elements": elements,
                "deferred_wall_p50_ms": deferred,
                "arena_wall_p50_ms": arena,
                "arena_graph_wall_p50_ms": graph,
                "arena_speedup": deferred / arena,
                "arena_graph_speedup": deferred / graph,
                "arena_capacity_bytes": int(max(
                    row["arena_capacity_bytes"] for row in grouped["arena"])),
                "expected_arena_capacity_bytes": expected_capacity,
                "arena_unique_addresses": int(max(
                    row["maximum_unique_addresses"] for row in grouped["arena"])),
                "arena_graph_unique_addresses": int(max(
                    row["maximum_unique_addresses"] for row in grouped["arena_graph"])),
                "arena_graph_node_count": int(max(
                    row["graph_node_count"] for row in grouped["arena_graph"])),
                "arena_graph_setup_ms": median("arena_graph", "graph_setup_ms"),
                "arena_graph_break_even_replays": math.ceil(
                    median("arena_graph", "graph_setup_ms") /
                    (deferred - graph)),
            })
    correctness = all(row["maximum_absolute_error"] == 0 for row in raw)
    layout = all(
        row["arena_capacity_bytes"] == row["expected_arena_capacity_bytes"] and
        row["arena_unique_addresses"] == 2 and
        row["arena_graph_unique_addresses"] == 2 and
        row["arena_graph_node_count"] == row["nodes"] + 1
        for row in comparisons)
    eager_keep = all(row["arena_speedup"] >= 1.05 for row in comparisons)
    graph_keep = all(row["arena_graph_speedup"] >= 1.05 for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and layout else "fail",
        "record_type": "activation_arena_matrix_summary",
        "raw_processes": len(raw),
        "correctness_gate": correctness,
        "layout_contract": layout,
        "arena_performance_gate": eager_keep,
        "arena_graph_performance_gate": graph_keep,
        "comparisons": comparisons,
        "decision": ("keep arena and arena Graph candidate"
                     if eager_keep and graph_keep else
                     "keep arena Graph; eager arena remains shape selective"
                     if graph_keep else
                     "keep eager arena; Graph remains shape selective"
                     if eager_keep else "reject arena policy"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in raw:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
