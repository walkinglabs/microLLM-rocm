#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-swiglu-compile"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_swiglu_compile_matrix.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_swiglu_compile.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 8
    assert {worker["first"] for worker in workers} == {
        "native", "eager", "compiled", "manual"}
    assert all(worker["status"] == "pass" and len(worker["records"]) == 2
               for worker in workers)
    assert all(worker["device_count_workaround"] ==
               "amdsmi_zero_fallback_to_hip_runtime" for worker in workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert summary["compiled_gate_pass"] is False
    assert summary["decision"] == "reject_compiled_swiglu"
    assert summary["worker_processes"] == 8 and summary["case_count"] == 2
    assert max(row["compiled_maximum_error"] for row in summary["groups"]) <= 4.8e-7
    assert max(row["compiled_loss_error"] for row in summary["groups"]) == 0.00390625
    assert max(row["compiled_vs_eager_event"] for row in summary["groups"]) <= 0.61
    assert max(row["compiled_vs_native_event"] for row in summary["groups"]) <= 0.48
    assert min(row["manual_vs_compiled_event"] for row in summary["groups"]) >= 7.69
    assert max(row["compile_cold_ms_median"] for row in summary["groups"]) >= 1160
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-swiglu-compile.svg")
    print("PyTorch SwiGLU compile rejection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

