#!/usr/bin/env python3
"""Subtract load+one-step Kernel stats from load+N-step stats."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys
from collections import defaultdict


def category(name: str) -> str:
    if name.startswith("Cijk_"):
        return "hipBLASLt GEMM"
    for needle, label in (
        ("adamw", "AdamW"),
        ("add_typed", "gradient/elementwise add"),
        ("cross_entropy", "cross entropy"),
        ("bias_gradient", "bias gradient"),
        ("cached_attention", "cached Attention"),
        ("kv_cache_store", "KV cache store"),
        ("strided_copy", "strided materialization"),
        ("cast_kernel", "FP32/BF16 cast"),
        ("softmax", "softmax"),
        ("repeat", "GQA repeat"),
        ("rms_norm", "RMSNorm forward/backward"),
        ("fill", "fill"),
    ):
        if needle in name:
            return label
    return "other kernels"


def read(path: pathlib.Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["Name"]: row for row in csv.DictReader(stream)}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-step", type=pathlib.Path, required=True)
    parser.add_argument("--many-step", type=pathlib.Path, required=True)
    parser.add_argument("--many-step-count", type=int, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument(
        "--track", default="training_kernel_phase_delta",
        choices=("training_kernel_phase_delta",
                 "inference_prefill_kernel_phase_delta",
                 "inference_cached_decode_kernel_phase_delta"))
    result = parser.parse_args()
    if result.many_step_count <= 1:
        parser.error("many-step-count must exceed one")
    return result


def main() -> int:
    args = options()
    one = read(args.one_step)
    many = read(args.many_step)
    divisor = args.many_step_count - 1
    kernels = []
    excluded = []
    negative_calls = []
    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name in sorted(set(one) | set(many)):
        row = many.get(name, {})
        one_row = one.get(name, {})
        call_delta = int(row.get("Calls", 0)) - int(one_row.get("Calls", 0))
        duration_delta = (int(row.get("TotalDurationNs", 0)) -
                    int(one_row.get("TotalDurationNs", 0))) / divisor
        if call_delta < 0:
            negative_calls.append(name)
            continue
        calls = call_delta / divisor
        duration = duration_delta
        if calls <= 0:
            excluded.append(name)
            continue
        if duration < 0:
            raise ValueError(
                f"positive-call Kernel has negative duration delta: {name}")
        label = category(name)
        kernels.append({
            "name": name,
            "category": label,
            "calls_per_step": calls,
            "duration_ns_per_step": duration,
        })
        grouped[label][0] += duration
        grouped[label][1] += calls
    kernels.sort(key=lambda row: row["duration_ns_per_step"], reverse=True)
    total = sum(row["duration_ns_per_step"] for row in kernels)
    if negative_calls:
        raise ValueError(
            "many-step profile has negative Kernel call deltas: " +
            ", ".join(negative_calls))
    if total <= 0:
        raise ValueError("profile delta contains no positive training Kernel time")
    categories = [
        {
            "category": label,
            "duration_ns_per_step": values[0],
            "calls_per_step": values[1],
            "kernel_share": values[0] / total if total else 0.0,
        }
        for label, values in grouped.items()
    ]
    categories.sort(key=lambda row: row["duration_ns_per_step"], reverse=True)
    result = {
        "schema_version": 1,
        "status": "pass",
        "track": args.track,
        "one_step_count": 1,
        "many_step_count": args.many_step_count,
        "derived_steps": divisor,
        "total_kernel_ns_per_step": total,
        "categories": categories,
        "top_kernels": kernels[:30],
        "raw_delta_kernel_names": len(set(one) | set(many)),
        "negative_call_delta_names": negative_calls,
        "excluded_nonpositive_delta_names": excluded,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.one_step, args.output_directory / "one-step-kernel-stats.csv")
    shutil.copy2(args.many_step, args.output_directory / "three-step-kernel-stats.csv")
    (args.output_directory / "profile-delta.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"profile_step_delta: {error}", file=sys.stderr)
        raise SystemExit(2)
