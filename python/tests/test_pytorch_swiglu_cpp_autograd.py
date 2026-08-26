#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-swiglu-cpp-autograd"


def main() -> int:
    comparer = ROOT / "benchmarks/single_gpu/compare_pytorch_swiglu_cpp_autograd.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_swiglu_cpp_autograd.py"
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
    assert comparison["decision"] == "recommend_cpp_autograd"
    assert comparison["correctness_pass"] is True
    assert comparison["speed_gate_pass"] is True
    assert comparison["memory_gate_pass"] is True
    assert comparison["fp32_native_gate_pass"] is True
    assert len(comparison["groups"]) == 6
    assert min(row["cpp_vs_python_event"] for row in comparison["groups"]) >= 1.28
    fp32 = [row for row in comparison["groups"] if row["dtype"] == "fp32"]
    assert min(row["cpp_vs_native_event"] for row in fp32) >= 1.13
    assert all(row["cpp_peak_extra_bytes"] == 1536 for row in fp32)
    low = [row for row in comparison["groups"] if row["dtype"] != "fp32"]
    assert all(row["cpp_peak_extra_bytes"] == row["native_peak_extra_bytes"]
               for row in low)
    cpp = (ROOT / "bindings/torch/torch_ops.cpp").read_text(encoding="utf-8")
    python = (ROOT / "python/microllm/torch_ops.py").read_text(encoding="utf-8")
    assert "class SwiGLUAutogradFunction" in cpp
    assert 'TORCH_LIBRARY_IMPL(microllm, Autograd' in cpp
    assert 'register_autograd(\n        "microllm::swiglu"' not in python
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-swiglu-cpp-autograd.svg")
    print("PyTorch C++ SwiGLU Autograd contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

