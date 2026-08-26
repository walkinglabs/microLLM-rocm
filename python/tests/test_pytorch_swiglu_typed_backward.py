#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-swiglu-typed-backward"


def main() -> int:
    comparer = ROOT / "benchmarks/single_gpu/compare_pytorch_swiglu_typed_backward.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_swiglu_typed_backward.py"
    ast.parse(comparer.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert all(worker["status"] == "pass" and len(worker["records"]) == 15
               for worker in workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert comparison["status"] == "pass"
    assert comparison["decision"] == "keep_typed_fused_backward"
    assert comparison["correctness_pass"] is True
    assert comparison["speed_gate_pass"] is True
    assert comparison["native_gate_pass"] is True
    assert comparison["memory_gate_pass"] is True
    assert len(comparison["groups"]) == 4
    assert min(row["typed_vs_aten_event"] for row in comparison["groups"]) >= 1.25
    assert min(row["typed_vs_native_event"] for row in comparison["groups"]) >= 1.047
    assert all(row["typed_peak_extra_bytes"] == row["native_peak_extra_bytes"]
               for row in comparison["groups"])
    bf16 = [row for row in comparison["groups"] if row["dtype"] == "bf16"]
    fp16 = [row for row in comparison["groups"] if row["dtype"] == "fp16"]
    assert all(row["maximum_error"] == 0.0 and row["maximum_rms_error"] == 0.0
               for row in bf16)
    assert max(row["maximum_error"] for row in fp16) <= 2.39e-7
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-swiglu-typed-backward.svg")
    print("PyTorch typed SwiGLU backward contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

