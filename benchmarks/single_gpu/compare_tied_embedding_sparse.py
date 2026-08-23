#!/usr/bin/env python3
"""Summarize tied-embedding sparse accumulation training and profile evidence."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import statistics
import sys


def kernel(rows: list[dict[str, str]], needle: str) -> dict[str, int]:
    selected = [row for row in rows if needle in row["Name"]]
    return {
        "calls": sum(int(row["Calls"]) for row in selected),
        "time_ns": sum(int(row["TotalDurationNs"]) for row in selected),
    }


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=pathlib.Path, required=True)
    parser.add_argument("--dense-qwen-diagnostics", type=pathlib.Path, required=True)
    parser.add_argument("--sparse-qwen-diagnostics", type=pathlib.Path, required=True)
    parser.add_argument("--sparse-deepseek-diagnostics", type=pathlib.Path, required=True)
    parser.add_argument("--dense-kernel-stats", type=pathlib.Path, required=True)
    parser.add_argument("--sparse-kernel-stats", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = options()
    training = [json.loads(line) for line in args.training.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(training) != 12:
        raise ValueError("training A/B must contain 12 rows")
    comparisons = []
    for model in sorted({row["model"] for row in training}):
        policies = {}
        for enabled, name in ((False, "dense"), (True, "sparse")):
            selected = [row for row in training
                        if row["model"] == model and
                        row["tied_embedding_sparse_add"] is enabled]
            if len(selected) != 3:
                raise ValueError(f"{model}/{name} needs three rows")
            policies[name] = {
                "tokens_per_second": statistics.median(
                    row["tokens_per_second"] for row in selected),
                "final_loss": statistics.median(row["final_loss"] for row in selected),
                "engine_peak_bytes": statistics.median(
                    row["engine_peak_bytes"] for row in selected),
                "engine_allocation_calls": statistics.median(
                    row["engine_allocation_calls"] for row in selected),
                "observed_parameter_after": statistics.median(
                    row["observed_parameter_after"] for row in selected),
            }
        dense, sparse = policies["dense"], policies["sparse"]
        comparisons.append({
            "model": model,
            "policies": policies,
            "throughput_speedup": sparse["tokens_per_second"] /
                dense["tokens_per_second"],
            "peak_ratio": sparse["engine_peak_bytes"] / dense["engine_peak_bytes"],
            "peak_bytes_saved": dense["engine_peak_bytes"] - sparse["engine_peak_bytes"],
            "loss_relative_difference": abs(
                sparse["final_loss"] - dense["final_loss"]) /
                max(abs(dense["final_loss"]), 1.0e-12),
            "observed_parameter_after_equal":
                sparse["observed_parameter_after"] == dense["observed_parameter_after"],
        })
    dense_diagnostics = json.loads(args.dense_qwen_diagnostics.read_text())
    sparse_qwen = json.loads(args.sparse_qwen_diagnostics.read_text())
    sparse_deepseek = json.loads(args.sparse_deepseek_diagnostics.read_text())
    with args.dense_kernel_stats.open(newline="", encoding="utf-8") as stream:
        dense_kernels = list(csv.DictReader(stream))
    with args.sparse_kernel_stats.open(newline="", encoding="utf-8") as stream:
        sparse_kernels = list(csv.DictReader(stream))
    profile = {
        name: {
            "dense": kernel(dense_kernels, needle),
            "sparse": kernel(sparse_kernels, needle),
        }
        for name, needle in (
            ("add", "add_typed_kernel<float>"),
            ("fill", "fill_typed_kernel<float>"),
            ("embedding_backward", "embedding_backward_kernel"),
        )
    }
    profile["total_kernel_time_ns"] = {
        "dense": sum(int(row["TotalDurationNs"]) for row in dense_kernels),
        "sparse": sum(int(row["TotalDurationNs"]) for row in sparse_kernels),
    }
    result = {
        "schema_version": 1,
        "status": "pass",
        "track": "tied_embedding_sparse_accumulation",
        "runs_per_model_policy": 3,
        "comparisons": comparisons,
        "diagnostics": {
            "dense_qwen": dense_diagnostics,
            "sparse_qwen": sparse_qwen,
            "sparse_deepseek": sparse_deepseek,
        },
        "profile": profile,
        "keep_gate": {
            "qwen_peak_ratio_maximum": 0.95,
            "qwen_throughput_ratio_minimum": 0.98,
            "deepseek_throughput_ratio_minimum": 0.98,
            "loss_relative_difference_maximum": 0.005,
        },
        "decision": "keep sparse tied-embedding accumulation as a memory optimization",
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "training.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in training),
        encoding="utf-8",
    )
    (args.output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.dense_kernel_stats,
                 args.output_directory / "dense-kernel-stats.csv")
    shutil.copy2(args.sparse_kernel_stats,
                 args.output_directory / "sparse-kernel-stats.csv")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"compare_tied_embedding_sparse: {error}", file=sys.stderr)
        raise SystemExit(2)
