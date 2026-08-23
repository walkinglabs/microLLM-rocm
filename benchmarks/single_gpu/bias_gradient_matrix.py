#!/usr/bin/env python3
"""Compare Scalar and cooperative bias-gradient kernels in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys


CASES = (
    (16, 896), (32, 128), (32, 256), (32, 896), (32, 1536),
    (64, 896), (128, 896), (256, 128),
    (512, 128), (512, 256), (512, 896), (512, 1536), (1024, 256),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    result = parser.parse_args()
    if result.runs <= 0 or result.warmup < 0 or result.repetitions <= 0:
        parser.error("runs/repetitions must be positive and warmup nonnegative")
    return result


def run(args: argparse.Namespace, rows: int, width: int,
        implementation: str, process_run: int) -> dict:
    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    command = [
        str(args.binary), "--rows", str(rows), "--width", str(width),
        "--implementation", implementation,
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
    ]
    completed = subprocess.run(
        command, env=environment, text=True, capture_output=True,
        timeout=args.timeout, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{rows}x{width}/{implementation}/run{process_run} failed: "
            f"{completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("bias-gradient benchmark must emit exactly one JSON row")
    result = json.loads(lines[0])
    result["process_run"] = process_run
    return result


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw = []
    summaries = []
    for rows, width in CASES:
        by_implementation = {}
        for implementation in ("scalar", "cooperative"):
            records = [run(args, rows, width, implementation, process_run + 1)
                       for process_run in range(args.runs)]
            raw.extend(records)
            by_implementation[implementation] = records
        scalar = statistics.median(
            row["event_ms_p50"] for row in by_implementation["scalar"])
        cooperative = statistics.median(
            row["event_ms_p50"] for row in by_implementation["cooperative"])
        summary = {
            "schema_version": 1,
            "status": "pass",
            "rows": rows,
            "width": width,
            "runs": args.runs,
            "scalar_event_ms_p50_median": scalar,
            "cooperative_event_ms_p50_median": cooperative,
            "cooperative_speedup": scalar / cooperative,
            "maximum_absolute_error": max(
                row["maximum_absolute_error"]
                for row in by_implementation["cooperative"]),
            "rms_error": max(
                row["rms_error"] for row in by_implementation["cooperative"]),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (args.output_directory / "operator-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "track": "cooperative_bias_gradient",
        "gpu": args.gpu,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "raw_rows": len(raw),
        "summaries": summaries,
    }
    (args.output_directory / "operator-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"bias_gradient_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
