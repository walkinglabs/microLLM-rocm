#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-swiglu"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_custom_op_swiglu_matrix.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_custom_op_swiglu.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert {row["order"] for row in workers} == {"torch-first", "microllm-first"}
    assert all(row["status"] == "pass" and len(row["records"]) == 15
               for row in workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert summary["case_count"] == 15 and summary["worker_processes"] == 6
    assert len(summary["groups"]) == 15
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["maximum_rms_error"] <= row["tolerance"] and
               row["maximum_loss_error"] <= row["tolerance"]
               for row in summary["groups"])
    forward_large = [row for row in summary["groups"]
                     if row["kind"] == "forward" and row["shape"] == "bandwidth"]
    backward_large = [row for row in summary["groups"]
                      if row["kind"] == "forward_backward" and row["shape"] == "large"]
    assert len(forward_large) == 3 and len(backward_large) == 3
    assert min(row["event_speedup_median"] for row in forward_large) >= 1.14
    assert max(row["event_speedup_median"] for row in forward_large) >= 1.57
    assert all(row["torch_peak_extra_bytes_median"] ==
               2 * row["microllm_peak_extra_bytes_median"] for row in forward_large)
    assert all(row["event_speedup_median"] < 0.77 for row in backward_large)
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-custom-op-swiglu.svg")
    print("PyTorch fused SwiGLU Custom Op contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

