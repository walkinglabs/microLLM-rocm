#!/usr/bin/env python3
"""Compare default, all-shape, and selective BF16 solution policies."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys


def read(path: pathlib.Path, registrations: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 6 or any(row.get("status") != "pass" or
                             row.get("bf16_algorithm_registrations") != registrations
                             for row in rows):
        raise ValueError(f"{path} does not match the expected policy")
    return rows


def summarize(rows: list[dict], model: str) -> dict:
    selected = [row for row in rows if row.get("model") == model]
    if len(selected) != 3:
        raise ValueError(f"{model} needs three runs")
    return {
        "tokens_per_second": statistics.median(
            row["tokens_per_second"] for row in selected),
        "final_loss": statistics.median(row["final_loss"] for row in selected),
        "engine_peak_bytes": statistics.median(
            row["engine_peak_bytes"] for row in selected),
        "observed_parameter_after": statistics.median(
            row["observed_parameter_after"] for row in selected),
    }


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--all-shapes", type=pathlib.Path, required=True)
    parser.add_argument("--selective", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = options()
    policies = {
        "baseline": read(args.baseline, 0),
        "all_shapes": read(args.all_shapes, 4),
        "selective": read(args.selective, 3),
    }
    models = sorted({row["model"] for row in policies["baseline"]})
    comparisons = []
    for model in models:
        rows = {name: summarize(records, model)
                for name, records in policies.items()}
        baseline = rows["baseline"]
        result = {"model": model, "policies": rows}
        for policy in ("all_shapes", "selective"):
            current = rows[policy]
            result[f"{policy}_speedup"] = (
                current["tokens_per_second"] / baseline["tokens_per_second"])
            result[f"{policy}_peak_ratio"] = (
                current["engine_peak_bytes"] / baseline["engine_peak_bytes"])
            result[f"{policy}_loss_relative_difference"] = abs(
                current["final_loss"] - baseline["final_loss"]) / max(
                    abs(baseline["final_loss"]), 1.0e-12)
            result[f"{policy}_parameter_equal"] = (
                current["observed_parameter_after"] ==
                baseline["observed_parameter_after"])
        comparisons.append(result)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    names = {"baseline": "training-baseline.jsonl",
             "all_shapes": "training-all-shapes.jsonl",
             "selective": "training-selective.jsonl"}
    for policy, rows in policies.items():
        (args.output_directory / names[policy]).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "status": "pass",
        "track": "bf16_solution_training_comparison",
        "runs_per_model_policy": 3,
        "comparisons": comparisons,
        "keep_gate": 1.05,
        "policies_passing_both_models": [],
        "decision": "discard model solution policy; retain explicit diagnostic tooling",
    }
    (args.output_directory / "training-comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"compare_bf16_solution_training: {error}", file=sys.stderr)
        raise SystemExit(2)
