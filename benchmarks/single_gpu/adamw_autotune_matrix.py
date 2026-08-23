#!/usr/bin/env python3
"""Run the correctness-first AdamW tuner in fresh processes and summarize medians."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any


DEFAULT_CASES = (
    {"name": "tail-4099-mirror", "elements": 4099, "mirror": True, "aligned": True},
    {"name": "tail-4099-unaligned", "elements": 4099, "mirror": False, "aligned": False},
    {"name": "qwen-mid-mirror", "elements": 802816, "mirror": True, "aligned": True},
    {"name": "qwen-embedding", "elements": 136134656, "mirror": False, "aligned": True},
    {"name": "deepseek-embedding", "elements": 233373696, "mirror": False, "aligned": True},
)


def implementation(row: dict[str, Any], name: str) -> dict[str, Any]:
    for candidate in row["candidates"]:
        if candidate["implementation"] == name:
            return candidate
    raise ValueError(f"missing {name} candidate")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty AdamW matrix")
    first = rows[0]
    for row in rows:
        for field in ("elements", "bf16_mirror", "aligned16", "architecture"):
            if row[field] != first[field]:
                raise ValueError(f"mixed {field} values in one case")
    scalar = [implementation(row, "scalar") for row in rows]
    vectorized = [implementation(row, "vectorized") for row in rows]
    scalar_p50 = statistics.median(candidate["event_ms_p50"] for candidate in scalar)
    vector_supported = all(candidate["supported"] for candidate in vectorized)
    vector_passed = all(candidate["correctness_passed"] for candidate in vectorized)
    vector_p50 = (
        statistics.median(candidate["event_ms_p50"] for candidate in vectorized)
        if vector_supported and vector_passed
        else 0.0
    )
    complete_state_passed = all(
        candidate["correctness_passed"] and candidate["finite"]
        for row in rows
        for candidate in row["candidates"]
        if candidate["supported"]
    )
    return {
        "schema_version": 1,
        "status": "pass" if complete_state_passed else "fail",
        "architecture": first["architecture"],
        "elements": first["elements"],
        "bf16_mirror": first["bf16_mirror"],
        "aligned16": first["aligned16"],
        "runs": len(rows),
        "complete_state_passed": complete_state_passed,
        "vectorized_supported": vector_supported,
        "vectorized_correctness_passed": vector_passed,
        "scalar_event_ms_p50_median": scalar_p50,
        "vectorized_event_ms_p50_median": vector_p50,
        "vectorized_speedup": scalar_p50 / vector_p50 if vector_p50 > 0.0 else 0.0,
        "recommendations": [row["recommended"] for row in rows],
    }


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> list[dict[str, Any]]:
    command = [
        str(args.binary),
        "--elements", str(case["elements"]),
        "--mirror", str(case["mirror"]).lower(),
        "--aligned", str(case["aligned"]).lower(),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
        "--mode", "training",
        "--accept", "false",
    ]
    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    rows = []
    for run in range(args.runs):
        process = subprocess.run(
            command, check=False, text=True, capture_output=True,
            env=environment, timeout=args.timeout,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"{case['name']} run {run} failed ({process.returncode}): "
                f"{process.stderr.strip()}"
            )
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"{case['name']} run {run} emitted {len(lines)} rows")
        row = json.loads(lines[0])
        row["case"] = case["name"]
        row["run"] = run
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cases", default=",".join(case["name"] for case in DEFAULT_CASES))
    result = parser.parse_args()
    if result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs/repetitions must be positive and warmup nonnegative")
    return result


def main() -> int:
    args = parse_args()
    requested = {name for name in args.cases.split(",") if name}
    available = {case["name"] for case in DEFAULT_CASES}
    if not requested or not requested <= available:
        raise ValueError(f"unknown case: {sorted(requested - available)}")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for case in DEFAULT_CASES:
        if case["name"] not in requested:
            continue
        rows = run_case(args, case)
        all_rows.extend(rows)
        summary = summarize(rows)
        summary["case"] = case["name"]
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "status": "pass" if all(row["status"] == "pass" for row in summaries) else "fail",
        "track": "adamw_correctness_before_timing_matrix",
        "gpu": args.gpu,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "summaries": summaries,
        "raw_rows": len(all_rows),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"adamw_autotune_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
