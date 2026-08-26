#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-softmax-attribution"


def main() -> int:
    for relative in (
        "benchmarks/single_gpu/pytorch_typed_softmax_attribution.py",
        "docs/optimization-log/scripts/render_pytorch_softmax_attribution.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(records) == 6 and summary["processes"] == 6
    assert summary["status"] == "pass"
    assert summary["maximum_raw_error"] <= 5.0e-4
    assert summary["maximum_cpp_error"] <= 5.0e-4
    assert summary["timed_payload_transfer_calls"] == 0
    assert 0.98 <= summary["cpp_over_raw_event_ratio"] <= 1.03
    assert summary["python_over_cpp_event_ratio"] >= 1.04
    assert 1.03 <= summary["raw_over_pytorch_event_ratio"] <= 1.08
    assert summary["python_over_pytorch_event_ratio"] >= 1.10
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-softmax-attribution.svg")
    print("PyTorch Softmax attribution contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
