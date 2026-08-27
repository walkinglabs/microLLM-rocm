#!/usr/bin/env python3
"""Pure name/layout mapping from Hugging Face Qwen parameters to microLLM."""

from __future__ import annotations


def internal_parameter(name: str) -> tuple[str, bool] | None:
    """Return the internal name and whether a rank-2 payload must transpose."""
    if name == "model.embed_tokens.weight":
        return "token_embedding.weight", False
    if name == "model.norm.weight":
        return "final_norm.weight", False
    if name == "lm_head.weight":
        return "output_head.weight", True
    fields = name.split(".")
    if len(fields) < 5 or fields[0] != "model" or fields[1] != "layers":
        return None
    layer = fields[2]
    suffix = ".".join(fields[3:])
    direct = {
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
    }
    if suffix in direct:
        return f"blocks.{layer}.{direct[suffix]}", False
    linear = {
        "self_attn.q_proj.weight": "attention.q_proj.weight",
        "self_attn.k_proj.weight": "attention.k_proj.weight",
        "self_attn.v_proj.weight": "attention.v_proj.weight",
        "self_attn.o_proj.weight": "attention.o_proj.weight",
        "mlp.gate_proj.weight": "feed_forward.gate_proj.weight",
        "mlp.up_proj.weight": "feed_forward.up_proj.weight",
        "mlp.down_proj.weight": "feed_forward.down_proj.weight",
    }
    if suffix in linear:
        return f"blocks.{layer}.{linear[suffix]}", True
    attention_bias = {
        "self_attn.q_proj.bias": "attention.q_proj.bias",
        "self_attn.k_proj.bias": "attention.k_proj.bias",
        "self_attn.v_proj.bias": "attention.v_proj.bias",
    }
    if suffix in attention_bias:
        return f"blocks.{layer}.{attention_bias[suffix]}", False
    return None
