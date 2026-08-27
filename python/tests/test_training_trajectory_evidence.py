#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "apps/hf_train_step.cpp"
COMPARE = ROOT / "apps/compare_safetensors.cpp"
PYTORCH = ROOT / "benchmarks/single_gpu/pytorch_hf_model_matrix.py"
MAPPING = ROOT / "benchmarks/single_gpu/hf_internal_parameter_mapping.py"
ALL_AUDIT = ROOT / "benchmarks/single_gpu/qwen3_training_all_parameter_audit.py"
ALL_RENDER = (ROOT / "docs/optimization-log/scripts" /
              "render_qwen3_training_all_parameter_audit.py")


def main() -> int:
    train = TRAIN.read_text(encoding="utf-8")
    compare = COMPARE.read_text(encoding="utf-8")
    pytorch = PYTORCH.read_text(encoding="utf-8")
    mapping_text = MAPPING.read_text(encoding="utf-8")
    all_audit = ALL_AUDIT.read_text(encoding="utf-8")
    all_render = ALL_RENDER.read_text(encoding="utf-8")
    for token in (
        "--loss-trajectory-output", "--gate-up-parameters-output",
        "--gate-up-gradients-output", "gate_up_gradient_tensors",
        "gate_up_gradient_elements",
        "write_loss_trajectory", "gate_up_parameter_tensors",
        "gate_up_parameter_elements", "--all-parameters-output",
        "--all-gradients-output", "all_gradient_tensors",
        "all_gradient_elements", "all_parameter_tensors",
        "all_parameter_elements", "all_state", "save_safetensors",
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
        "--all-parameters-output", "--all-gradients-output",
        "internal_parameter", "internal_state", "save_gate_up", "save_all",
        "gate_up_gradient_tensors", "all_gradient_tensors",
    ):
        assert token in pytorch
    for token in (
        '"lm_head.weight"', '"self_attn.q_proj.bias"',
        '"self_attn.k_proj.bias"', '"self_attn.v_proj.bias"',
        '"self_attn.o_proj.weight"', '"mlp.down_proj.weight"',
    ):
        assert token in mapping_text
    specification = importlib.util.spec_from_file_location("hf_mapping", MAPPING)
    assert specification is not None and specification.loader is not None
    mapping = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(mapping)
    cases = {
        "model.embed_tokens.weight": ("token_embedding.weight", False),
        "model.norm.weight": ("final_norm.weight", False),
        "lm_head.weight": ("output_head.weight", True),
        "model.layers.7.input_layernorm.weight":
            ("blocks.7.attention_norm.weight", False),
        "model.layers.7.post_attention_layernorm.weight":
            ("blocks.7.ffn_norm.weight", False),
        "model.layers.7.self_attn.q_norm.weight":
            ("blocks.7.attention.q_norm.weight", False),
        "model.layers.7.self_attn.k_norm.weight":
            ("blocks.7.attention.k_norm.weight", False),
        "model.layers.7.self_attn.q_proj.weight":
            ("blocks.7.attention.q_proj.weight", True),
        "model.layers.7.self_attn.k_proj.weight":
            ("blocks.7.attention.k_proj.weight", True),
        "model.layers.7.self_attn.v_proj.weight":
            ("blocks.7.attention.v_proj.weight", True),
        "model.layers.7.self_attn.o_proj.weight":
            ("blocks.7.attention.o_proj.weight", True),
        "model.layers.7.self_attn.q_proj.bias":
            ("blocks.7.attention.q_proj.bias", False),
        "model.layers.7.self_attn.k_proj.bias":
            ("blocks.7.attention.k_proj.bias", False),
        "model.layers.7.self_attn.v_proj.bias":
            ("blocks.7.attention.v_proj.bias", False),
        "model.layers.7.mlp.gate_proj.weight":
            ("blocks.7.feed_forward.gate_proj.weight", True),
        "model.layers.7.mlp.up_proj.weight":
            ("blocks.7.feed_forward.up_proj.weight", True),
        "model.layers.7.mlp.down_proj.weight":
            ("blocks.7.feed_forward.down_proj.weight", True),
    }
    for source, expected in cases.items():
        assert mapping.internal_parameter(source) == expected
    assert mapping.internal_parameter("model.layers.7.unknown.weight") is None
    for token in (
        "EXPECTED_TENSORS = 310", "EXPECTED_ELEMENTS = 596_049_920",
        "EXPECTED_STORED_TENSORS = 311", "parameter_family",
        "attention_qkv", "ffn_gate_up", "--all-gradients-output",
        "--all-parameters-output", "tied_alias", "gradient_families",
        "parameter_families", "temporary_exports_removed",
    ):
        assert token in all_audit
    for token in (
        "310 independent Tensors", "FP32 · PASS", "BF16 · REJECT",
        "gradient_families", "fixed 5e-2 gate", "temporary 9.54 GB exports removed",
    ):
        assert token in all_render
    print("training trajectory evidence contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
