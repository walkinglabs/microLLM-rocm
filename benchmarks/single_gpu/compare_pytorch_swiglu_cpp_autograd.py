#!/usr/bin/env python3
"""Compare Python-registered and C++ SwiGLU Autograd matrices."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
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
    options = arguments()
    baseline = raw(options.baseline)
    candidate = raw(options.candidate)
    groups = []
    for dtype in ("fp32", "fp16", "bf16"):
        for shape, elements in (("medium", 65536), ("large", 1 << 20)):
            old = [row for row in baseline if row["kind"] == "forward_backward" and
                   row["dtype"] == dtype and row["shape"] == shape]
            new = [row for row in candidate if row["kind"] == "forward_backward" and
                   row["dtype"] == dtype and row["shape"] == shape]
            old_event = statistics.median(row["microllm_event_ms"] for row in old)
            new_event = statistics.median(row["microllm_event_ms"] for row in new)
            groups.append({
                "dtype": dtype, "shape": shape, "elements": elements,
                "cpp_vs_python_event": old_event / new_event,
                "cpp_vs_native_event": statistics.median(
                    row["torch_event_ms"] / row["microllm_event_ms"] for row in new),
                "python_peak_extra_bytes": statistics.median(
                    row["microllm_peak_extra_bytes"] for row in old),
                "cpp_peak_extra_bytes": statistics.median(
                    row["microllm_peak_extra_bytes"] for row in new),
                "native_peak_extra_bytes": statistics.median(
                    row["torch_peak_extra_bytes"] for row in new),
                "maximum_error": max(row["maximum_error"] for row in new),
                "maximum_rms_error": max(row["rms_error"] for row in new),
                "tolerance": new[0]["tolerance"],
            })
    correctness = all(row["maximum_error"] <= row["tolerance"] and
                      row["maximum_rms_error"] <= row["tolerance"] for row in groups)
    speed = all(row["cpp_vs_python_event"] >= 1.1 for row in groups)
    memory = all(row["cpp_peak_extra_bytes"] <= row["python_peak_extra_bytes"]
                 for row in groups)
    fp32_native = all(row["cpp_vs_native_event"] >= 1.05
                      for row in groups if row["dtype"] == "fp32")
    report = {
        "schema_version": 1,
        "status": "pass" if correctness and speed and memory and fp32_native else "fail",
        "record_type": "pytorch_rocm_swiglu_cpp_autograd_comparison",
        "correctness_pass": correctness, "speed_gate_pass": speed,
        "memory_gate_pass": memory, "fp32_native_gate_pass": fp32_native,
        "decision": ("recommend_cpp_autograd"
                     if correctness and speed and memory and fp32_native
                     else "reject_cpp_autograd"),
        "groups": groups,
    }
    options.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

