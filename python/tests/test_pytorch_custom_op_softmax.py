#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax"
CURRENT = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax-inference-gate"


def summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_custom_op_softmax.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_custom_op_softmax.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    baseline = summary(BASELINE)
    current = summary(CURRENT)
    assert baseline["worker_processes"] == current["worker_processes"] == 6
    assert baseline["case_count"] == current["case_count"] == 10
    assert current["status"] == "pass" and current["correctness_pass"] is True
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_outputs_distinct"] and
               row["custom_peak_extra_bytes_median"] ==
               row["native_peak_extra_bytes_median"]
               for row in current["groups"])
    before = {(row["dtype"], row["width"]): row for row in baseline["groups"]}
    after = {(row["dtype"], row["width"]): row for row in current["groups"]}
    fp16_wide = ("fp16", 4096)
    assert (before[fp16_wide]["custom_event_ms_median"] /
            after[fp16_wide]["custom_event_ms_median"]) >= 1.10
    assert after[("fp16", 1024)]["event_speedup_median"] >= 0.98
    assert after[("bf16", 1024)]["event_speedup_median"] >= 0.98
    assert after[fp16_wide]["event_speedup_median"] >= 0.78
    source = (ROOT / "bindings/torch/torch_ops.cpp").read_text(encoding="utf-8")
    assert 'library.def("softmax(Tensor input) -> Tensor")' in source
    assert "!input.requires_grad()" in source
    assert 'library.impl("softmax", &softmax_autograd)' in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-custom-op-softmax.svg")
    print("PyTorch Custom Op Softmax contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
