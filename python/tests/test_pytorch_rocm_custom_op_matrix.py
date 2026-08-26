#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-ops"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_custom_op_rocm_matrix.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_rocm_custom_ops.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert {row["order"] for row in workers} == {"torch-first", "microllm-first"}
    assert {row["run"] for row in workers} == {1, 2, 3}
    assert all(row["status"] == "pass" and len(row["records"]) == 20
               for row in workers)
    assert summary["status"] == "pass"
    assert summary["correctness_pass"] is True
    assert summary["worker_processes"] == 6
    assert summary["case_count"] == 20
    assert summary["architecture"] == "gfx942"
    assert summary["torch_hip_version"] == "7.13.99004"
    assert len(summary["groups"]) == 20
    assert {row["dtype"] for row in summary["groups"]} == {"fp32", "fp16", "bf16"}
    assert {row["kind"] for row in summary["groups"]} == {
        "forward", "forward_backward"}
    assert all(row["processes"] == 6 for row in summary["groups"])
    assert all(row["maximum_error"] == 0.0 and
               row["maximum_rms_error"] == 0.0 and
               row["maximum_loss_error"] == 0.0 for row in summary["groups"])
    assert all(row["torch_peak_extra_bytes_median"] ==
               row["microllm_peak_extra_bytes_median"] for row in summary["groups"])
    assert all(row["event_speedup_median"] < 1.0 for row in summary["groups"])
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-custom-ops.svg")
    print("PyTorch ROCm Custom Op matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

