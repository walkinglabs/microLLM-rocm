#!/usr/bin/env python3
"""Compare pre/post scalar-seed fused SwiGLU Autograd matrices."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def raw(directory: Path) -> list[dict]:
    workers = [json.loads(line) for line in
               (directory / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for worker in workers for row in worker["records"]]


def main() -> int:
    options = args()
    baseline = raw(options.baseline)
    candidate = raw(options.candidate)
    groups = []
    for shape, elements in (("medium", 65536), ("large", 1 << 20)):
        before = [row for row in baseline if row["kind"] == "forward_backward" and
                  row["dtype"] == "fp32" and row["shape"] == shape]
        after = [row for row in candidate if row["kind"] == "forward_backward" and
                 row["dtype"] == "fp32" and row["shape"] == shape]
        before_event = statistics.median(row["microllm_event_ms"] for row in before)
        after_event = statistics.median(row["microllm_event_ms"] for row in after)
        before_peak = statistics.median(row["microllm_peak_extra_bytes"] for row in before)
        after_peak = statistics.median(row["microllm_peak_extra_bytes"] for row in after)
        groups.append({
            "shape": shape, "elements": elements,
            "candidate_vs_baseline_event": before_event / after_event,
            "baseline_peak_extra_bytes": before_peak,
            "candidate_peak_extra_bytes": after_peak,
            "peak_reduction_fraction": 1.0 - after_peak / before_peak,
            "maximum_error": max(row["maximum_error"] for row in after),
            "maximum_rms_error": max(row["rms_error"] for row in after),
            "native_event_ratio": statistics.median(
                row["torch_event_ms"] / row["microllm_event_ms"] for row in after),
        })
    correctness = all(row["maximum_error"] <= 3.0e-6 for row in groups)
    memory = all(row["peak_reduction_fraction"] >= 0.99 for row in groups)
    performance = all(row["candidate_vs_baseline_event"] >= 0.98 for row in groups)
    report = {
        "schema_version": 1,
        "status": "pass" if correctness and memory and performance else "fail",
        "record_type": "pytorch_rocm_swiglu_scalar_seed_comparison",
        "correctness_pass": correctness,
        "memory_gate_pass": memory,
        "performance_non_regression_pass": performance,
        "decision": ("keep_scalar_seed_route"
                     if correctness and memory and performance
                     else "reject_scalar_seed_route"),
        "groups": groups,
    }
    options.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

