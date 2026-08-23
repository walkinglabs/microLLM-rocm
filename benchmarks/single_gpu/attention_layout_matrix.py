#!/usr/bin/env python3
"""Run a fresh-process matrix for interleaved-head Attention P*V."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


DEFAULT_SHAPES = (
    "edge:2:2:3:2",
    "short:2:14:32:64",
    "medium:1:14:128:64",
    "qwen_t512:1:14:512:64",
    "deepseek_t512:1:12:512:128",
)


def parse_shape(specification: str) -> dict:
    fields = specification.split(":")
    if len(fields) != 5:
        raise ValueError("shape must be name:B:H:T:D")
    name = fields[0]
    values = [int(field) for field in fields[1:]]
    if not name or any(value <= 0 for value in values):
        raise ValueError("shape name and dimensions must be positive")
    return dict(zip(("name", "batch", "heads", "sequence", "width"),
                    (name, *values), strict=True))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--shape", action="append", dest="shapes")
    result = parser.parse_args()
    if not result.binary.is_file():
        parser.error(f"benchmark binary does not exist: {result.binary}")
    if result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs/repetitions must be positive and warmup nonnegative")
    result.shapes = [parse_shape(item) for item in
                     (result.shapes or DEFAULT_SHAPES)]
    return result


def command(args: argparse.Namespace, shape: dict,
            implementation: str) -> list[str]:
    return [
        str(args.binary),
        "--batch", str(shape["batch"]),
        "--heads", str(shape["heads"]),
        "--sequence", str(shape["sequence"]),
        "--width", str(shape["width"]),
        "--implementation", implementation,
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ]


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    records = []
    for process_run in range(1, args.runs + 1):
        for shape in args.shapes:
            implementations = ("materialized", "interleaved") \
                if process_run % 2 else ("interleaved", "materialized")
            for implementation in implementations:
                completed = subprocess.run(
                    command(args, shape, implementation), check=True, text=True,
                    capture_output=True)
                record = json.loads(completed.stdout)
                if (record.get("status") != "pass" or
                        not record.get("finite") or
                        record.get("host_to_device_calls") != 0 or
                        record.get("device_to_host_calls") != 0 or
                        float(record["maximum_absolute_error"]) > 3.0e-4 or
                        float(record["rms_error"]) > 1.0e-5):
                    raise RuntimeError(
                        f"{shape['name']}/{implementation} failed a correctness gate")
                record.update({"shape_name": shape["name"],
                               "process_run": process_run})
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    rows = []
    for shape in args.shapes:
        selected = [record for record in records
                    if record["shape_name"] == shape["name"]]
        implementations = {}
        for implementation in ("materialized", "interleaved"):
            group = [record for record in selected
                     if record["implementation"] == implementation]
            if len(group) != args.runs:
                raise RuntimeError(
                    f"{shape['name']}/{implementation} has incomplete processes")
            implementations[implementation] = {
                "event_ms_p50": median(group, "event_ms_p50"),
                "event_ms_p95": median(group, "event_ms_p95"),
                "wall_ms_p50": median(group, "wall_ms_p50"),
                "maximum_absolute_error": max(
                    float(row["maximum_absolute_error"]) for row in group),
                "rms_error": max(float(row["rms_error"]) for row in group),
            }
        baseline = implementations["materialized"]
        candidate = implementations["interleaved"]
        rows.append({
            **shape,
            "implementations": implementations,
            "event_speedup": baseline["event_ms_p50"] /
                candidate["event_ms_p50"],
            "wall_speedup": baseline["wall_ms_p50"] /
                candidate["wall_ms_p50"],
        })
    summary = {
        "schema_version": 1,
        "status": "pass",
        "track": "attention_interleaved_probability_value",
        "runs_per_shape_implementation": args.runs,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rows": rows,
        "gate_results": {
            "complete_outputs": all(
                row["implementations"]["interleaved"]["maximum_absolute_error"]
                <= 3.0e-4 and
                row["implementations"]["interleaved"]["rms_error"] <= 1.0e-5
                for row in rows),
            "official_t512_speedup": all(
                row["event_speedup"] >= 1.05 for row in rows
                if row["name"].endswith("t512")),
        },
    }
    if not all(summary["gate_results"].values()):
        summary["status"] = "reject"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
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
        print(f"attention_layout_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
