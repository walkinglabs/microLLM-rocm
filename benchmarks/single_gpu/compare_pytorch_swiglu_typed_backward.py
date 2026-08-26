#!/usr/bin/env python3
"""Compare C++ ATen and fused typed SwiGLU backward matrices."""

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
    for dtype in ("fp16", "bf16"):
        for shape, elements in (("medium", 65536), ("large", 1 << 20)):
            old = [row for row in baseline if row["kind"] == "forward_backward" and
                   row["dtype"] == dtype and row["shape"] == shape]
            new = [row for row in candidate if row["kind"] == "forward_backward" and
                   row["dtype"] == dtype and row["shape"] == shape]
            groups.append({
                "dtype": dtype, "shape": shape, "elements": elements,
                "typed_vs_aten_event": statistics.median(
                    row["microllm_event_ms"] for row in old) /
                    statistics.median(row["microllm_event_ms"] for row in new),
                "typed_vs_native_event": statistics.median(
                    row["torch_event_ms"] / row["microllm_event_ms"] for row in new),
                "aten_peak_extra_bytes": statistics.median(
                    row["microllm_peak_extra_bytes"] for row in old),
                "typed_peak_extra_bytes": statistics.median(
                    row["microllm_peak_extra_bytes"] for row in new),
                "native_peak_extra_bytes": statistics.median(
                    row["torch_peak_extra_bytes"] for row in new),
                "maximum_error": max(row["maximum_error"] for row in new),
                "maximum_rms_error": max(row["rms_error"] for row in new),
                "tolerance": new[0]["tolerance"],
            })
    correctness = all(row["maximum_error"] <= row["tolerance"] and
                      row["maximum_rms_error"] <= row["tolerance"] for row in groups)
    speed = all(row["typed_vs_aten_event"] >= 1.2 for row in groups)
    native = all(row["typed_vs_native_event"] >= 1.03 for row in groups)
    memory = all(row["typed_peak_extra_bytes"] == row["native_peak_extra_bytes"] and
                 row["typed_peak_extra_bytes"] <= row["aten_peak_extra_bytes"]
                 for row in groups)
    report = {
        "schema_version": 1,
        "status": "pass" if correctness and speed and native and memory else "fail",
        "record_type": "pytorch_rocm_swiglu_typed_backward_comparison",
        "correctness_pass": correctness, "speed_gate_pass": speed,
        "native_gate_pass": native, "memory_gate_pass": memory,
        "decision": ("keep_typed_fused_backward"
                     if correctness and speed and native and memory
                     else "reject_typed_fused_backward"),
        "groups": groups,
    }
    options.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

