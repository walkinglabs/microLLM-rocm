#!/usr/bin/env python3
"""Compare scalar, broad-vector, and selective-vector PyTorch Custom Op evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--selective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def raw_records(directory: Path) -> list[dict]:
    workers = [json.loads(line) for line in
               (directory / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    return [record for worker in workers for record in worker["records"]]


def key(record: dict) -> tuple:
    return (record["kind"], record["operation"], record["dtype"],
            record["shape"], record["elements"])


def medians(records: list[dict]) -> dict[tuple, dict]:
    result = {}
    for record_key in sorted({key(row) for row in records}):
        selected = [row for row in records if key(row) == record_key]
        result[record_key] = {
            "microllm_event_ms": statistics.median(
                row["microllm_event_ms"] for row in selected),
            "torch_event_ms": statistics.median(
                row["torch_event_ms"] for row in selected),
            "maximum_error": max(row["maximum_error"] for row in selected),
            "maximum_rms_error": max(row["rms_error"] for row in selected),
            "maximum_loss_error": max(row.get("loss_error", 0.0) for row in selected),
            "peak_equal": all(row["microllm_peak_extra_bytes"] ==
                              row["torch_peak_extra_bytes"] for row in selected),
        }
    return result


def main() -> int:
    args = arguments()
    baseline = medians(raw_records(args.baseline))
    broad = medians(raw_records(args.broad))
    selective = medians(raw_records(args.selective))
    if baseline.keys() != broad.keys() or baseline.keys() != selective.keys():
        raise RuntimeError("matrix keys differ")
    groups = []
    for record_key in baseline:
        base = baseline[record_key]
        wide = broad[record_key]
        chosen = selective[record_key]
        groups.append({
            "kind": record_key[0], "operation": record_key[1],
            "dtype": record_key[2], "shape": record_key[3],
            "elements": record_key[4],
            "broad_vs_scalar_event": (
                base["microllm_event_ms"] / wide["microllm_event_ms"]),
            "selective_vs_scalar_event": (
                base["microllm_event_ms"] / chosen["microllm_event_ms"]),
            "scalar_torch_event_drift": (
                base["torch_event_ms"] / chosen["torch_event_ms"]),
            "maximum_error": chosen["maximum_error"],
            "maximum_rms_error": chosen["maximum_rms_error"],
            "maximum_loss_error": chosen["maximum_loss_error"],
            "peak_equal": chosen["peak_equal"],
        })
    low_bandwidth = [row for row in groups if row["kind"] == "forward" and
                     row["dtype"] in ("fp16", "bf16") and
                     row["shape"] == "bandwidth"]
    fp32_bandwidth = [row for row in groups if row["kind"] == "forward" and
                      row["dtype"] == "fp32" and row["shape"] == "bandwidth"]
    correctness = all(row["maximum_error"] == 0.0 and
                      row["maximum_rms_error"] == 0.0 and
                      row["maximum_loss_error"] == 0.0 and row["peak_equal"]
                      for row in groups)
    low_precision_gate = all(row["selective_vs_scalar_event"] >= 1.05
                             for row in low_bandwidth)
    fp32_non_regression = all(row["selective_vs_scalar_event"] >= 0.95
                             for row in fp32_bandwidth)
    broad_fp32_rejected = any(row["broad_vs_scalar_event"] < 0.95
                              for row in fp32_bandwidth)
    admitted = correctness and low_precision_gate and fp32_non_regression and \
        broad_fp32_rejected
    report = {
        "schema_version": 1,
        "status": "pass" if admitted else "fail",
        "record_type": "pytorch_rocm_custom_op_vector16_comparison",
        "correctness_pass": correctness,
        "low_precision_gate_pass": low_precision_gate,
        "fp32_non_regression_pass": fp32_non_regression,
        "broad_fp32_rejected": broad_fp32_rejected,
        "decision": ("keep_selective_low_precision_vector16"
                     if admitted else "reject_vector16"),
        "low_precision_minimum_speedup": min(
            row["selective_vs_scalar_event"] for row in low_bandwidth),
        "low_precision_maximum_speedup": max(
            row["selective_vs_scalar_event"] for row in low_bandwidth),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())

