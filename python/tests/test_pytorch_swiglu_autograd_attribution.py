#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-swiglu-autograd-attribution"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_swiglu_autograd_attribution.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_swiglu_autograd_attribution.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert {worker["first"] for worker in workers} == {"native", "custom", "manual"}
    assert all(worker["status"] == "pass" and len(worker["records"]) == 2
               for worker in workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert summary["worker_processes"] == 6 and summary["case_count"] == 2
    assert all(max(row["custom_maximum_error"], row["manual_maximum_error"],
                   row["custom_loss_error"], row["manual_loss_error"]) <= 4.8e-7
               for row in summary["groups"])
    assert min(row["manual_vs_custom_event"] for row in summary["groups"]) >= 4.85
    assert min(row["manual_vs_native_event"] for row in summary["groups"]) >= 3.85
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-swiglu-autograd-attribution.svg")
    print("PyTorch SwiGLU Autograd attribution contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

