#!/usr/bin/env python3
"""Combine phase-differential profiles for the BTHD BF16 Q/K experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b")
BLOCKS = {"qwen2.5-0.5b": 24, "deepseek-r1-distill-qwen-1.5b": 28}


def category(document: dict, name: str) -> dict:
    return next(row for row in document["categories"] if row["category"] == name)


def rope_time(document: dict) -> float:
    return sum(float(row["duration_ns_per_step"])
               for row in document["top_kernels"]
               if "rope_split_half_bias_bthd_kernel" in row["name"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-summary", required=True, type=Path)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    performance = json.loads(args.performance_summary.read_text(encoding="utf-8"))
    rows = []
    for model in MODELS:
        baseline = json.loads((args.profile_root / model / "fp32-boundary" /
                               "profile-delta.json").read_text(encoding="utf-8"))
        candidate = json.loads((args.profile_root / model / "bf16-qk" /
                                "profile-delta.json").read_text(encoding="utf-8"))
        before_cast = category(baseline, "FP32/BF16 cast")
        after_cast = category(candidate, "FP32/BF16 cast")
        rows.append({
            "model": model,
            "baseline_total_kernel_ns": baseline["total_kernel_ns_per_step"],
            "bf16_qk_total_kernel_ns": candidate["total_kernel_ns_per_step"],
            "total_kernel_speedup": baseline["total_kernel_ns_per_step"] /
                                    candidate["total_kernel_ns_per_step"],
            "baseline_cast_calls": before_cast["calls_per_step"],
            "bf16_qk_cast_calls": after_cast["calls_per_step"],
            "cast_calls_removed": before_cast["calls_per_step"] -
                                  after_cast["calls_per_step"],
            "expected_cast_calls_removed": BLOCKS[model] * 2,
            "baseline_cast_ns": before_cast["duration_ns_per_step"],
            "bf16_qk_cast_ns": after_cast["duration_ns_per_step"],
            "cast_ns_saved": before_cast["duration_ns_per_step"] -
                             after_cast["duration_ns_per_step"],
            "baseline_rope_ns": rope_time(baseline),
            "bf16_qk_rope_ns": rope_time(candidate),
        })
    call_gate = all(row["cast_calls_removed"] == row["expected_cast_calls_removed"]
                    for row in rows)
    kernel_gate = all(row["total_kernel_speedup"] > 1.0 for row in rows)
    result = {
        "schema_version": 1, "status": "pass" if call_gate and kernel_gate else "fail",
        "record_type": "inference_bthd_bf16_qk_profile_summary",
        "profile_processes": 8, "derived_forwards": 20,
        "cast_elimination_gate": call_gate, "kernel_performance_gate": kernel_gate,
        "formal_performance_gate": performance.get("performance_gate"),
        "comparisons": rows,
        "decision": performance.get("decision"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
