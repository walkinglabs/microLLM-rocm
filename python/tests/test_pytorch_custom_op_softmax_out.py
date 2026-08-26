#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax-out"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_custom_op_softmax_out.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_custom_op_softmax_out.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert summary["status"] == "pass" and summary["case_count"] == 10
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_returned_pointers_match"] and row["all_peak_extra_zero"]
               for row in summary["groups"])
    groups = {(row["dtype"], row["width"]): row for row in summary["groups"]}
    assert groups[("fp16", 1024)]["event_speedup_median"] >= 1.05
    assert groups[("bf16", 1024)]["event_speedup_median"] >= 1.05
    assert groups[("fp16", 4096)]["event_speedup_median"] >= 0.80
    source = (ROOT / "bindings/torch/torch_ops.cpp").read_text(encoding="utf-8")
    assert 'Tensor(a!) output) -> Tensor(a!)' in source
    assert "softmax_out is inference-only" in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-custom-op-softmax-out.svg")
    print("PyTorch Custom Op Softmax out contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
