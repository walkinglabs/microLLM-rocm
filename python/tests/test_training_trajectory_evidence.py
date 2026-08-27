#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "apps/hf_train_step.cpp"
COMPARE = ROOT / "apps/compare_safetensors.cpp"
PYTORCH = ROOT / "benchmarks/single_gpu/pytorch_hf_model_matrix.py"


def main() -> int:
    train = TRAIN.read_text(encoding="utf-8")
    compare = COMPARE.read_text(encoding="utf-8")
    pytorch = PYTORCH.read_text(encoding="utf-8")
    for token in (
        "--loss-trajectory-output", "--gate-up-parameters-output",
        "--gate-up-gradients-output", "gate_up_gradient_tensors",
        "gate_up_gradient_elements",
        "write_loss_trajectory", "gate_up_parameter_tensors",
        "gate_up_parameter_elements", "save_safetensors",
    ):
        assert token in train
    assert "--bf16-gate-up-weight-gradient" not in train
    for token in (
        "load_safetensors", "safetensors_complete_comparison",
        "maximum_absolute_difference", "rms_difference",
        "compared_elements", "all_finite",
    ):
        assert token in compare
    for token in (
        "--gate-up-parameters-output", "--gate-up-gradients-output",
        "gate_up_state", "save_gate_up", "gate_up_gradient_tensors",
    ):
        assert token in pytorch
    print("training trajectory evidence contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
