#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-swiglu-scalar-seed"


def main() -> int:
    comparer = ROOT / "benchmarks/single_gpu/compare_pytorch_swiglu_scalar_seed.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_swiglu_scalar_seed.py"
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
    assert comparison["decision"] == "keep_scalar_seed_route"
    assert comparison["correctness_pass"] is True
    assert comparison["memory_gate_pass"] is True
    assert comparison["performance_non_regression_pass"] is True
    assert len(comparison["groups"]) == 2
    assert min(row["candidate_vs_baseline_event"]
               for row in comparison["groups"]) >= 1.08
    assert min(row["peak_reduction_fraction"]
               for row in comparison["groups"]) >= 0.994
    assert all(row["candidate_peak_extra_bytes"] == 1536
               for row in comparison["groups"])
    bridge = (ROOT / "python/microllm/torch_ops.py").read_text(encoding="utf-8")
    for token in ("all(", "stride == 0", "gradient.as_strided((1,), (0,))",
                  "swiglu_backward_scalar_seed"):
        assert token in bridge
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-swiglu-scalar-seed.svg")
    print("PyTorch SwiGLU scalar-seed contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

