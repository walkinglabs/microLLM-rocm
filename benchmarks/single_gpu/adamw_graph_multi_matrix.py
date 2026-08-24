#!/usr/bin/env python3
"""Measure stable-descriptor multi-tensor AdamW HIP Graph replay."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


PRECISIONS = ("fp32", "bf16")
MODES = ("eager", "graph", "graph-multi")


def cases(value: str) -> list[tuple[int, int]]:
    result = []
    try:
        for item in value.split(","):
            tensors, elements = item.lower().split("x", 1)
            result.append((int(tensors), int(elements)))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must use TENSORSxELEMENTS comma syntax") from error
    if not result or len(set(result)) != len(result) or \
       any(tensors <= 0 or elements <= 0 for tensors, elements in result):
        raise argparse.ArgumentTypeError("cases must be unique and positive")
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument(
        "--cases", type=cases,
        default=cases("1x1024,16x1024,64x1024,256x1024,16x262144"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=50)
    result = parser.parse_args()
    if not result.binary.is_file():
        parser.error(f"binary does not exist: {result.binary}")
    if result.runs < 3 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs must be at least three and timing counts valid")
    return result


def expected_nodes(mode: str, tensors: int) -> int:
    if mode == "eager": return 0
    if mode == "graph": return tensors + 1
    return 2


def execute(args: argparse.Namespace, precision: str, mode: str,
            tensors: int, elements: int) -> dict:
    completed = subprocess.run([
        str(args.binary), "--precision", precision, "--mode", mode,
        "--tensors", str(tensors), "--elements", str(elements),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("AdamW Graph benchmark did not emit one JSON object") from error
    if row.get("schema_version") != 1 or row.get("status") != "pass" or \
       row.get("record_type") != "adamw_graph_replay_measurement" or \
       row.get("precision") != precision or row.get("mode") != mode or \
       row.get("tensors") != tensors or row.get("elements") != elements or \
       row.get("captured_nodes") != expected_nodes(mode, tensors) or \
       row.get("final_step") != args.warmup + args.repetitions or \
       any(row.get(field) != 0 for field in (
           "timed_host_to_device_calls", "timed_device_to_host_calls",
           "timed_device_to_device_calls")) or \
       not math.isfinite(float(row.get("event_ms_per_step", math.nan))) or \
       not math.isfinite(float(row.get("wall_ms_per_step", math.nan))) or \
       not math.isfinite(float(row.get("preparation_ms", math.nan))):
        raise RuntimeError(
            f"invalid multi Graph row: {precision}/{mode}/{tensors}x{elements}")
    return row


def maximum_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right): return math.inf
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)),
               default=0.0)


def summarize(records: list[dict], case_values: list[tuple[int, int]],
              runs: int) -> dict:
    comparisons = []
    for precision in PRECISIONS:
        for tensors, elements in case_values:
            selected = [row for row in records
                        if row["precision"] == precision and
                        row["tensors"] == tensors and
                        row["elements"] == elements]
            policies = {}
            for mode in MODES:
                rows = [row for row in selected if row["mode"] == mode]
                if len(rows) != runs:
                    raise RuntimeError(
                        f"incomplete multi Graph case: {precision}/{tensors}x{elements}/{mode}")
                policies[mode] = {
                    "event_ms_per_step": statistics.median(
                        float(row["event_ms_per_step"]) for row in rows),
                    "wall_ms_per_step": statistics.median(
                        float(row["wall_ms_per_step"]) for row in rows),
                    "preparation_ms": statistics.median(
                        float(row["preparation_ms"]) for row in rows),
                    "setup_ms": statistics.median(
                        float(row["setup_ms"]) for row in rows),
                }
            errors = {"graph": [], "graph-multi": []}
            for process_run in range(1, runs + 1):
                eager = next(row for row in selected
                             if row["mode"] == "eager" and
                             row["process_run"] == process_run)
                for mode in ("graph", "graph-multi"):
                    candidate = next(row for row in selected
                                     if row["mode"] == mode and
                                     row["process_run"] == process_run)
                    errors[mode].extend(
                        maximum_error(eager[field], candidate[field])
                        for field in (
                            "parameter_sample", "first_moment_sample",
                            "second_moment_sample", "mirror_sample"))
            eager_wall = policies["eager"]["wall_ms_per_step"]
            graph_wall = policies["graph"]["wall_ms_per_step"]
            multi_wall = policies["graph-multi"]["wall_ms_per_step"]
            comparisons.append({
                "precision": precision,
                "tensors": tensors,
                "elements": elements,
                "runs": runs,
                "policies": policies,
                "per_tensor_maximum_state_error": max(errors["graph"]),
                "multi_maximum_state_error": max(errors["graph-multi"]),
                "per_tensor_wall_speedup": eager_wall / graph_wall,
                "multi_wall_speedup": eager_wall / multi_wall,
                "multi_vs_per_tensor": graph_wall / multi_wall,
            })
    bf16_many = [row for row in comparisons
                 if row["precision"] == "bf16" and
                 row["tensors"] >= 64 and row["elements"] == 1024]
    large = [row for row in comparisons if row["elements"] == 262144]
    gates = {
        "both_graph_paths_align_complete_state_samples": all(
            row["per_tensor_maximum_state_error"] <= 2.0e-6 and
            row["multi_maximum_state_error"] <= 2.0e-6
            for row in comparisons),
        "multi_graph_has_exactly_two_nodes": all(
            row["captured_nodes"] == expected_nodes(
                row["mode"], row["tensors"])
            for row in records),
        "timed_region_has_no_descriptor_or_payload_transfer": all(
            row["timed_host_to_device_calls"] == 0 and
            row["timed_device_to_host_calls"] == 0 and
            row["timed_device_to_device_calls"] == 0
            for row in records),
        "bf16_many_small_rescued": all(
            row["multi_wall_speedup"] >= 1.05 for row in bf16_many),
        "large_tensor_rescued": all(
            row["multi_wall_speedup"] >= 1.0 for row in large),
        "multi_beats_per_tensor_every_case": all(
            row["multi_vs_per_tensor"] >= 1.05 for row in comparisons),
    }
    required = (
        "both_graph_paths_align_complete_state_samples",
        "multi_graph_has_exactly_two_nodes",
        "timed_region_has_no_descriptor_or_payload_transfer",
    )
    return {
        "schema_version": 1,
        "status": "pass" if all(gates[name] for name in required) else "fail",
        "experiment": "stable_descriptor_adamw_multi_graph",
        "processes": len(records),
        "runs_per_mode_case": runs,
        "comparisons": comparisons,
        "gates": gates,
        "decision": (
            "keep explicit two-node multi-tensor Graph candidate"
            if gates["bf16_many_small_rescued"] else
            "reject multi-tensor Graph candidate"),
    }


def main() -> int:
    args = options()
    records = []
    base = tuple((precision, tensors, elements)
                 for precision in PRECISIONS
                 for tensors, elements in args.cases)
    for process_run in range(1, args.runs + 1):
        ordered = base if process_run % 2 else tuple(reversed(base))
        for precision, tensors, elements in ordered:
            modes = MODES if process_run % 2 else tuple(reversed(MODES))
            for mode in modes:
                row = execute(args, precision, mode, tensors, elements)
                row["process_run"] = process_run
                records.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    summary = summarize(records, args.cases, args.runs)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"adamw_graph_multi_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
