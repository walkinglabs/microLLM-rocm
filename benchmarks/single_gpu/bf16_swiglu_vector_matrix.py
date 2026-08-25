#!/usr/bin/env python3
"""Repeated-process BF16 SwiGLU scalar/vector operator gate."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


CASES = (
    ("qwen2.5-0.5b", 1024 * 4864),
    ("deepseek-r1-distill-qwen-1.5b", 1024 * 8960),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    result = parser.parse_args()
    if (not result.binary.is_file() or result.runs <= 0 or
            result.warmup < 0 or result.repetitions <= 0):
        parser.error("runner arguments are outside the measured contract")
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("benchmark did not emit a JSON object")


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    records = []
    for model, elements in CASES:
        for process_run in range(1, args.runs + 1):
            order = ("scalar", "vectorized")
            if process_run % 2 == 0:
                order = tuple(reversed(order))
            for implementation in order:
                command = [
                    str(args.binary), "--op", "swiglu", "--device", "hip",
                    "--dtype", "bf16", "--implementation", implementation,
                    "--size", str(elements), "--warmup", str(args.warmup),
                    "--repetitions", str(args.repetitions),
                ]
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False)
                stem = f"{model}-p{process_run}-{implementation}"
                (logs / f"{stem}.stdout.txt").write_text(
                    completed.stdout, encoding="utf-8")
                (logs / f"{stem}.stderr.txt").write_text(
                    completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{stem} failed with exit {completed.returncode}")
                row = last_json(completed.stdout)
                if (row.get("op") != "swiglu" or
                        row.get("implementation") != implementation or
                        int(row.get("size", 0)) != elements):
                    raise RuntimeError(f"{stem} violated the benchmark schema")
                row.update({"model": model, "process_run": process_run})
                records.append(row)

    comparisons = []
    for model, elements in CASES:
        selected = [row for row in records if row["model"] == model]
        scalar = [float(row["kernel_ms_mean"]) for row in selected
                  if row["implementation"] == "scalar"]
        vectorized = [float(row["kernel_ms_mean"]) for row in selected
                      if row["implementation"] == "vectorized"]
        maximum_error = max(float(row["maximum_absolute_error"])
                            for row in selected)
        scalar_median = statistics.median(scalar)
        vector_median = statistics.median(vectorized)
        speedup = scalar_median / vector_median
        comparisons.append({
            "model": model,
            "sequence": 1024,
            "elements": elements,
            "runs_per_implementation": args.runs,
            "scalar_kernel_ms_mean_median": scalar_median,
            "vectorized_kernel_ms_mean_median": vector_median,
            "speedup": speedup,
            "maximum_absolute_error": maximum_error,
            "correctness_passed": maximum_error == 0.0,
            "performance_passed": speedup >= 1.05,
        })
    keep = all(row["correctness_passed"] and row["performance_passed"]
               for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "track": "bf16_swiglu_vector_operator",
        "raw_processes": len(records),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "comparisons": comparisons,
        "operator_gate_passed": keep,
        "decision": "admit full-model gate" if keep else "reject vector route",
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
