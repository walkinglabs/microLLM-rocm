#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_weight_gradient_models.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--bf16-gate-up-weight-gradient", "1048576",
        "bf16_gate_up_weight_gradient_assignments", "48", "56",
        "alternating policy order", "throughput_speedup_minimum",
        "peak_ratio_maximum", "final_loss_relative_difference_maximum",
        "first_loss_relative_difference_maximum",
        "observed_parameter_relative_difference_maximum",
        '"training.jsonl"',
        "keep explicit candidate for longer training validation",
    ):
        assert token in text
    print("BF16 weight-gradient model gate contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
