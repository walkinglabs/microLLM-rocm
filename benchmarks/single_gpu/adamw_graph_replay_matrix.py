#!/usr/bin/env python3
"""Compare eager and device-stepped HIP Graph AdamW in fresh processes."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


PRECISIONS = ("fp32", "bf16")
MODES = ("eager", "graph")


def cases(value: str) -> list[tuple[int, int]]:
    result = []
    try:
        for item in value.split(","):
            tensors, elements = item.lower().split("x", 1)
            result.append((int(tensors), int(elements)))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must use TENSORSxELEMENTS comma syntax") from error
    if not result or any(tensors <= 0 or elements <= 0
                         for tensors, elements in result) or \
       len(set(result)) != len(result):
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
       row.get("final_step") != args.warmup + args.repetitions or \
       any(int(row.get(field, -1)) != 0 for field in (
           "timed_host_to_device_calls", "timed_device_to_host_calls",
           "timed_device_to_device_calls")) or \
       not math.isfinite(float(row.get("event_ms_per_step", math.nan))) or \
       not math.isfinite(float(row.get("wall_ms_per_step", math.nan))):
        raise RuntimeError(f"invalid AdamW Graph row: {precision}/{mode}/{tensors}x{elements}")
    expected_nodes = tensors + 1 if mode == "graph" else 0
    if row.get("captured_nodes") != expected_nodes:
        raise RuntimeError("AdamW Graph captured-node contract changed")
    return row


def maximum_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
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
                        f"incomplete AdamW Graph case: {precision}/{tensors}x{elements}/{mode}")
                policies[mode] = {
                    "event_ms_per_step": statistics.median(
                        float(row["event_ms_per_step"]) for row in rows),
                    "wall_ms_per_step": statistics.median(
                        float(row["wall_ms_per_step"]) for row in rows),
                    "setup_ms": statistics.median(
                        float(row["setup_ms"]) for row in rows),
                }
            errors = []
            for process_run in range(1, runs + 1):
                eager = next(row for row in selected
                             if row["mode"] == "eager" and
                             row["process_run"] == process_run)
                graph = next(row for row in selected
                             if row["mode"] == "graph" and
                             row["process_run"] == process_run)
                errors.extend(maximum_error(eager[field], graph[field])
                              for field in (
                                  "parameter_sample", "first_moment_sample",
                                  "second_moment_sample", "mirror_sample"))
            comparisons.append({
                "precision": precision,
                "tensors": tensors,
                "elements": elements,
                "runs": runs,
                "policies": policies,
                "maximum_state_error": max(errors, default=math.inf),
                "event_speedup": (
                    policies["eager"]["event_ms_per_step"] /
                    policies["graph"]["event_ms_per_step"]),
                "wall_speedup": (
                    policies["eager"]["wall_ms_per_step"] /
                    policies["graph"]["wall_ms_per_step"]),
            })
    fp32_small_many = [row for row in comparisons
                       if row["precision"] == "fp32" and
                       row["tensors"] >= 64 and row["elements"] == 1024]
    bf16_small_many = [row for row in comparisons
                       if row["precision"] == "bf16" and
                       row["tensors"] >= 64 and row["elements"] == 1024]
    large = [row for row in comparisons if row["elements"] == 262144]
    gates = {
        "complete_state_samples_align": all(
            row["maximum_state_error"] <= 2.0e-6 for row in comparisons),
        "graph_nodes_and_device_step_are_exact": all(
            row["captured_nodes"] ==
                (row["tensors"] + 1 if row["mode"] == "graph" else 0) and
            row["final_step"] == row["warmup"] + row["repetitions"]
            for row in records),
        "timed_region_has_no_payload_transfers": all(
            row["timed_host_to_device_calls"] == 0 and
            row["timed_device_to_host_calls"] == 0 and
            row["timed_device_to_device_calls"] == 0
            for row in records),
        "fp32_many_small_tensors_wall_speedup_at_least_1_05": all(
            row["wall_speedup"] >= 1.05 for row in fp32_small_many),
        "bf16_many_small_tensors_wall_speedup_at_least_1_05": all(
            row["wall_speedup"] >= 1.05 for row in bf16_small_many),
        "large_tensor_universal_policy": all(
            row["wall_speedup"] >= 1.0 for row in large),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(gates[name] for name in (
            "complete_state_samples_align",
            "graph_nodes_and_device_step_are_exact",
            "timed_region_has_no_payload_transfers")) else "fail",
        "experiment": "device_owned_adamw_graph_replay",
        "processes": len(records),
        "runs_per_mode_case": runs,
        "comparisons": comparisons,
        "gates": gates,
        "decision": (
            "keep explicit device-step Graph primitive and FP32 many-small candidate; "
            "reject universal and BF16 routing"),
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
        print(f"adamw_graph_replay_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
