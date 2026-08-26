#!/usr/bin/env python3
"""Aggregate raw launcher, C++ op, Python C API and PyTorch Softmax timings."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--samples", type=int, default=7)
    return parser.parse_args()


def median(rows: list[dict], field: str) -> float:
    return statistics.median(row[field] for row in rows)


def main() -> int:
    options = arguments()
    if options.runs <= 0:
        raise ValueError("runs must be positive")
    records = []
    for run in range(1, options.runs + 1):
        for order in ("raw-first", "cpp-first"):
            command = [
                str(options.binary), "--rows", "8", "--width", "4096",
                "--warmup", str(options.warmup),
                "--repetitions", str(options.repetitions),
                "--samples", str(options.samples), "--order", order,
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"attribution worker failed: {completed.stderr}\n{completed.stdout}")
            record = json.loads(completed.stdout)
            record["run"] = run
            records.append(record)

    python_workers = [json.loads(line) for line in
                      (options.python_results / "raw.jsonl")
                      .read_text(encoding="utf-8").splitlines()]
    python_rows = [row for worker in python_workers for row in worker["records"]
                   if row["dtype"] == "fp16" and row["width"] == 4096]
    summary = {
        "schema_version": 1,
        "record_type": "typed_softmax_attribution_matrix",
        "status": "pass",
        "processes": len(records),
        "rows": 8,
        "width": 4096,
        "raw_event_ms_median": median(records, "raw_event_ms"),
        "raw_wall_ms_median": median(records, "raw_wall_ms"),
        "cpp_event_ms_median": median(records, "cpp_event_ms"),
        "cpp_wall_ms_median": median(records, "cpp_wall_ms"),
        "python_capi_event_ms_median": median(python_rows, "microllm_event_ms"),
        "python_capi_wall_ms_median": median(python_rows, "microllm_wall_ms"),
        "pytorch_event_ms_median": median(python_rows, "torch_event_ms"),
        "pytorch_wall_ms_median": median(python_rows, "torch_wall_ms"),
        "maximum_raw_error": max(row["raw_maximum_error"] for row in records),
        "maximum_cpp_error": max(row["cpp_maximum_error"] for row in records),
        "timed_payload_transfer_calls": sum(
            row["timed_h2d_calls"] + row["timed_d2h_calls"]
            for row in records),
    }
    summary["cpp_over_raw_event_ratio"] = (
        summary["cpp_event_ms_median"] / summary["raw_event_ms_median"])
    summary["python_over_cpp_event_ratio"] = (
        summary["python_capi_event_ms_median"] / summary["cpp_event_ms_median"])
    summary["raw_over_pytorch_event_ratio"] = (
        summary["raw_event_ms_median"] / summary["pytorch_event_ms_median"])
    summary["python_over_pytorch_event_ratio"] = (
        summary["python_capi_event_ms_median"] / summary["pytorch_event_ms_median"])
    if (summary["maximum_raw_error"] > 5.0e-4 or
            summary["maximum_cpp_error"] > 5.0e-4 or
            summary["timed_payload_transfer_calls"] != 0):
        summary["status"] = "fail"

    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (options.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
