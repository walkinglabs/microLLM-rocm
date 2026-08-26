#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax-out-wave1024"
CANDIDATE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax-out-fallthrough"


def rows(path: Path) -> dict[tuple[str, int], dict]:
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert data["status"] == "pass" and data["case_count"] == 10
    return {(row["dtype"], row["width"]): row for row in data["groups"]}


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_softmax_out_fallthrough_reject.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    before, after = rows(BASELINE), rows(CANDIDATE)
    for dtype in ("fp16", "bf16"):
        key = (dtype, 4096)
        gain = (before[key]["custom_event_ms_median"] /
                after[key]["custom_event_ms_median"])
        assert gain < 1.05
        assert after[key]["maximum_error"] <= after[key]["tolerance"]
        assert after[key]["all_returned_pointers_match"]
        assert after[key]["all_peak_extra_zero"]
    source = (ROOT / "bindings/torch/torch_ops.cpp").read_text(encoding="utf-8")
    assert 'library.impl("softmax_out", &softmax_out_autograd)' in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-softmax-out-fallthrough-reject.svg")
    print("PyTorch Softmax-out fallthrough rejection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
