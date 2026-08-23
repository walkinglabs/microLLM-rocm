#!/usr/bin/env python3
"""Validate and summarize same-revision Scalar/cooperative training rows."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys


FIELDS = (
    "tokens_per_second", "first_loss", "final_loss", "engine_peak_bytes",
    "observed_parameter_before", "observed_parameter_after",
    "mean_optimizer_ms", "mean_step_ms",
)


def read(path: pathlib.Path, expected_policy: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 6 or any(row.get("status") != "pass" or
                             row.get("bias_gradient_policy") != expected_policy
                             for row in rows):
        raise ValueError(f"{path} is not a complete three-run/two-model matrix")
    return rows


def medians(rows: list[dict], model: str) -> dict:
    selected = [row for row in rows if row.get("model") == model]
    if len(selected) != 3 or sorted(row.get("process_run") for row in selected) != [1, 2, 3]:
        raise ValueError(f"{model} does not have exactly three process runs")
    return {field: statistics.median(float(row[field]) for row in selected)
            for field in FIELDS}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = options()
    baseline = read(args.baseline, "scalar_columns")
    candidate = read(args.candidate, "cooperative_rows_32")
    models = sorted({row["model"] for row in baseline})
    if models != sorted({row["model"] for row in candidate}):
        raise ValueError("baseline and candidate model sets differ")
    comparisons = []
    for model in models:
        before = medians(baseline, model)
        after = medians(candidate, model)
        comparison = {
            "model": model,
            "baseline": before,
            "candidate": after,
            "throughput_speedup": after["tokens_per_second"] /
                before["tokens_per_second"],
            "optimizer_speedup": before["mean_optimizer_ms"] /
                after["mean_optimizer_ms"],
            "peak_ratio": after["engine_peak_bytes"] / before["engine_peak_bytes"],
            "first_loss_absolute_difference": abs(
                after["first_loss"] - before["first_loss"]),
            "final_loss_absolute_difference": abs(
                after["final_loss"] - before["final_loss"]),
            "final_loss_relative_difference": abs(
                after["final_loss"] - before["final_loss"]) /
                max(abs(before["final_loss"]), 1.0e-12),
            "observed_parameter_after_equal":
                after["observed_parameter_after"] == before["observed_parameter_after"],
        }
        if comparison["throughput_speedup"] < 1.05 or \
                comparison["peak_ratio"] != 1.0 or \
                comparison["final_loss_relative_difference"] > 0.005 or \
                not comparison["observed_parameter_after_equal"]:
            raise ValueError(f"{model} failed the cooperative bias-gradient keep gate")
        comparisons.append(comparison)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "baseline.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in baseline),
        encoding="utf-8",
    )
    (args.output_directory / "candidate.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "track": "cooperative_bias_gradient_training",
        "baseline_policy": "scalar_columns",
        "candidate_policy": "cooperative_rows_32",
        "runs_per_model_policy": 3,
        "comparisons": comparisons,
    }
    (args.output_directory / "training-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"compare_bias_gradient_training: {error}", file=sys.stderr)
        raise SystemExit(2)
