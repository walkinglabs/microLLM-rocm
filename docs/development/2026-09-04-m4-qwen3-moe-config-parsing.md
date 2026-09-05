# 2026-09-04 — M4 Qwen3 MoE config parsing

## Contract

Parse `num_experts`, `num_experts_per_tok`, `norm_topk_prob`, and
`moe_intermediate_size` from a `model_type=qwen3_moe` Hugging Face config, and
explicitly reject config shapes this repo does not implement rather than silently
ignoring them. This milestone is parsing only — no weight loading, no
model-level forward, no parameter counting.

## What was added

- `ModelConfig` (`include/microllm/model/config.h`) gained four fields:
  `moe_num_experts`, `moe_num_experts_per_tok`, `moe_intermediate_size`,
  `moe_norm_topk_prob`. All-zero/false means dense (no MoE); `validate()` enforces
  an all-or-nothing invariant — `moe_num_experts == 0` requires the other two
  numeric fields to also be zero, and `moe_num_experts > 0` requires
  `0 < moe_num_experts_per_tok <= moe_num_experts` and a positive
  `moe_intermediate_size`.
- `load_huggingface_config` (`src/model/huggingface.cpp`) now accepts
  `model_type=qwen3_moe` alongside the existing `qwen2`/`qwen3`. It shares Qwen3's
  attention treatment (`head_dim`, `qk_norm`) since Qwen3 MoE's attention block is
  architecturally identical to dense Qwen3 — only the FFN block is replaced by
  routed experts.
- Three fields are explicitly rejected rather than silently dropped when present
  on a `qwen3_moe` config, each for a documented reason (see the inline comment
  in `huggingface.cpp` and the "MoE 路由" section of
  `docs/OPERATOR_CONTRACTS.zh-CN.md`):
  - `decoder_sparse_step != 1` or a non-empty `mlp_only_layers` — both express a
    per-layer dense/MoE mix; this parser only represents "every layer is MoE."
  - `router_aux_loss_coef`'s presence at all — it configures a training-time
    load-balancing loss this repo does not implement. Silently ignoring it would
    let a caller believe the field was honored.

## An assumption worth flagging

The MoE field names (`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
`norm_topk_prob`, `decoder_sparse_step`, `mlp_only_layers`, `router_aux_loss_coef`)
follow the naming convention Hugging Face's Qwen2MoeConfig already uses, which
Qwen3MoeConfig is expected to follow since Qwen3 is a direct evolution of Qwen2's
config schema. This machine has no network access to fetch a real
`Qwen3-30B-A3B`-style `config.json` to confirm the field names byte-for-byte
against transformers' actual `Qwen3MoeConfig` source. All tests here use synthetic
JSON fixtures with these assumed names, not a pinned real checkpoint config, unlike
the existing `ParsesPinnedQwen3ExplicitHeadAndQkNormContract` test. **The first time
a real Qwen3 MoE `config.json` is available, re-check these field names before
trusting this parser against it** — this is exactly what M7 ("end-to-end
real-checkpoint gate") exists to close.

The `mlp_only_layers` non-empty check is also a simplification: it compares the
raw JSON substring to the literal string `"[]"`, so a real config using different
whitespace (e.g. `"[ ]"`) would be misclassified as non-empty and rejected when it
should be accepted. This repository's JSON parser has no general array-value
reader; adding one only for this one field felt like more machinery than the
milestone needed. Revisit if a real config trips this.

## Verification

```text
CPU Debug (MICROLLM_ENABLE_HIP=OFF): 295/295 microllm_tests passed (was 292 after
M3; +3 new: ModelConfigTest.MoeFieldsAreExplicitAndConsistent,
HuggingFaceConfigTest.ParsesQwen3MoeConfigAndRejectsParameterCounting,
HuggingFaceConfigTest.RejectsUnsupportedQwen3MoeFields).
scripts/audit_test_coverage.py: pass (unaffected -- config.h/huggingface.h are not
part of the audited ops.h/autograd.h public-symbol surface).
```

`parameter_count()`/`weight_bytes()` explicitly `throw std::invalid_argument` for
any config with `moe_num_experts != 0`, rather than silently returning a
dense-only (wrong) count. Working out the exact per-expert parameter/tensor count
is deferred to M5, where it can be checked against a real checkpoint's tensor
manifest rather than a formula nobody has verified.

## Current boundary

No weight loading (`state_dict`/`load_state_dict` don't know about per-expert
tensor names yet), no model-level forward, no CLI/inference integration. The next
milestone (M5) adds per-expert weight naming (e.g.
`blocks.N.moe.experts.E.gate_proj.weight`) and strict-mode loading with an exact
total-tensor-count assertion.
