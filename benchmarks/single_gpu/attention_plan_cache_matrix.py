#!/usr/bin/env python3
"""Compare cached and uncached interleaved Attention plans in fresh processes."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys


DEFAULT_SHAPES = (
    "edge:2:2:3:2",
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


def command(args: argparse.Namespace, shape: dict, cached: bool) -> list[str]:
    return [
        str(args.binary),
        "--batch", str(shape["batch"]),
        "--heads", str(shape["heads"]),
        "--sequence", str(shape["sequence"]),
        "--width", str(shape["width"]),
        "--implementation", "interleaved",
        "--plan-cache", "true" if cached else "false",
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
            policies = (False, True) if process_run % 2 else (True, False)
            for cached in policies:
                completed = subprocess.run(
                    command(args, shape, cached), check=True, text=True,
                    capture_output=True)
                record = json.loads(completed.stdout)
                expected_entries = 1 if cached else 0
                if (record.get("status") != "pass" or
                        record.get("plan_cache") is not cached or
                        record.get("plan_cache_entries") != expected_entries or
                        float(record.get("maximum_absolute_error", 1)) > 3.0e-4 or
                        float(record.get("rms_error", 1)) > 1.0e-5 or
                        record.get("host_to_device_calls") != 0 or
                        record.get("device_to_host_calls") != 0):
                    raise RuntimeError(
                        f"{shape['name']}/{cached} failed a correctness/cache gate")
                if cached and (record.get("plan_cache_misses") != 1 or
                               record.get("plan_cache_hits", 0) <
                               args.warmup + args.repetitions):
                    raise RuntimeError(f"{shape['name']} did not reuse its plan")
                if not cached and (record.get("plan_cache_hits") != 0 or
                                   record.get("plan_cache_misses") != 0):
                    raise RuntimeError(f"{shape['name']} uncached route mutated cache stats")
                record.update({"shape_name": shape["name"],
                               "process_run": process_run,
                               "policy": "cached" if cached else "uncached"})
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    rows = []
    for shape in args.shapes:
        selected = [row for row in records
                    if row["shape_name"] == shape["name"]]
        policies = {}
        for policy in ("uncached", "cached"):
            group = [row for row in selected if row["policy"] == policy]
            if len(group) != args.runs:
                raise RuntimeError(f"{shape['name']}/{policy} is incomplete")
            policies[policy] = {
                "event_ms_p50": median(group, "event_ms_p50"),
                "wall_ms_p50": median(group, "wall_ms_p50"),
                "event_ms_p95": median(group, "event_ms_p95"),
                "wall_ms_p95": median(group, "wall_ms_p95"),
            }
        rows.append({
            **shape,
            "policies": policies,
            "event_speedup": policies["uncached"]["event_ms_p50"] /
                policies["cached"]["event_ms_p50"],
            "wall_speedup": policies["uncached"]["wall_ms_p50"] /
                policies["cached"]["wall_ms_p50"],
        })
    summary = {
        "schema_version": 1,
        "status": "pass",
        "track": "attention_interleaved_plan_cache",
        "runs_per_shape_policy": args.runs,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rows": rows,
        "gate_results": {
            "official_t512_wall_speedup": all(
                row["wall_speedup"] >= 1.01 for row in rows
                if row["name"].endswith("t512")),
        },
    }
    if not all(summary["gate_results"].values()):
        summary["status"] = "reject"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "operator-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "operator-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            subprocess.CalledProcessError, RuntimeError) as error:
        print(f"attention_plan_cache_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
