#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_weight_gradient_trajectory.py"
TRAIN = ROOT / "apps/hf_train_step.cpp"
COMPARE = ROOT / "apps/compare_safetensors.cpp"


def main() -> int:
    runner = RUNNER.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    compare = COMPARE.read_text(encoding="utf-8")
    ast.parse(runner)
    for token in (
        "--loss-trajectory-output", "--gate-up-parameters-output",
        "safetensors_complete_comparison", "loss_relative_difference_maximum",
        "parameter_maximum_absolute_difference", "parameter_rms_difference",
        "admit default gate/up BF16 weight gradients", '"trajectory.jsonl"',
    ):
        assert token in runner
    for token in (
        "write_loss_trajectory", "gate_up_parameter_tensors",
        "gate_up_parameter_elements", "save_safetensors",
    ):
        assert token in train
    for token in (
        "load_safetensors", "maximum_absolute_difference", "rms_difference",
        "compared_elements", "all_finite",
    ):
        assert token in compare
    print("BF16 weight-gradient trajectory contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

