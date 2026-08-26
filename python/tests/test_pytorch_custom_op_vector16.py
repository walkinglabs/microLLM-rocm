#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks/results"
SCALAR = BASE / "2026-08-26-pytorch-rocm-custom-ops"
BROAD = BASE / "2026-08-26-pytorch-rocm-custom-ops-vector16"
SELECTIVE = BASE / "2026-08-26-pytorch-rocm-custom-ops-vector16-selective"


def main() -> int:
    comparer = ROOT / "benchmarks/single_gpu/compare_pytorch_custom_op_vector16.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_custom_op_vector16.py"
    ast.parse(comparer.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    for directory in (SCALAR, BROAD, SELECTIVE):
        workers = [json.loads(line) for line in
                   (directory / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert len(workers) == 6
        assert all(worker["status"] == "pass" and len(worker["records"]) == 20
                   for worker in workers)
        assert summary["correctness_pass"] is True and summary["case_count"] == 20
    report = json.loads((SELECTIVE / "comparison.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["decision"] == "keep_selective_low_precision_vector16"
    assert report["correctness_pass"] is True
    assert report["low_precision_gate_pass"] is True
    assert report["fp32_non_regression_pass"] is True
    assert report["broad_fp32_rejected"] is True
    assert report["low_precision_minimum_speedup"] >= 1.27
    assert report["low_precision_maximum_speedup"] >= 1.40
    assert len(report["groups"]) == 20
    assert all(row["maximum_error"] == 0.0 and
               row["maximum_rms_error"] == 0.0 and
               row["maximum_loss_error"] == 0.0 and row["peak_equal"]
               for row in report["groups"])
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-custom-op-vector16.svg")
    print("PyTorch Custom Op vector16 contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

