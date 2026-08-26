#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-swiglu-backward"


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_swiglu_backward.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert all(worker["status"] == "pass" and len(worker["records"]) == 4
               for worker in workers)
    assert summary["status"] == "pass"
    assert summary["correctness_pass"] is True
    assert summary["performance_gate_pass"] is False
    assert summary["decision"] == "reject_vectorized_backward"
    assert summary["case_count"] == 4 and summary["worker_processes"] == 6
    assert all(row["maximum_error"] <= 1.2e-7 and
               row["scalar_vector_maximum_error"] <= 3.0e-8
               for row in summary["groups"])
    assert all(row["vector_vs_scalar_event_median"] < 1.05
               for row in summary["groups"])
    assert min(row["vector_vs_native_event_median"]
               for row in summary["groups"]) >= 2.0
    assert all(row["scalar_peak_extra_bytes_median"] ==
               row["vector_peak_extra_bytes_median"] and
               row["native_peak_extra_bytes_median"] * 2 ==
               row["scalar_peak_extra_bytes_median"] * 3
               for row in summary["groups"])
    source = (ROOT / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    public = (ROOT / "include/microllm/ops/ops.h").read_text(encoding="utf-8")
    assert "swiglu_backward_float4_kernel" not in source
    assert "SwiGLUBackwardImplementation" not in public
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-swiglu-backward.svg")
    print("PyTorch SwiGLU backward rejection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

