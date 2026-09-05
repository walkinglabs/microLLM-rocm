# 2026-09-04 — M6 Qwen3 MoE model-level full-graph gate

## Contract

Give `TransformerModel::forward()` (and the graph-free inference path) a real
MoE computation, replacing `MoeFeedForward`'s M5 throws, and gate it with the
actual, unmodified HF Transformers Qwen3MoE forward as oracle — not a
hand-rolled equivalent — at `atol=2e-3, rtol=2e-3`.

## M5's internal weight layout was wrong; corrected before writing any forward code

Before writing this milestone's oracle, `transformers` was checked for a real
`Qwen3MoeConfig`/`Qwen3MoeExperts` implementation (version 5.8.0, available in
a local conda env — see the M4/M5 records for why this hadn't been checked
earlier). It revealed that M5's internal MoE weight representation — one
separate `Linear` per expert per projection (`~3×num_experts` tensors per
layer) — does not match how a real Qwen3-MoE checkpoint is actually shaped.
The real `Qwen3MoeExperts` module stores exactly two packed parameters per
layer, regardless of expert count:

- `gate_up_proj`: `[num_experts, 2*moe_intermediate_size, hidden_size]` — gate
  and up fused together (nn.Linear-style, output-dimension first), chunked at
  forward time;
- `down_proj`: `[num_experts, hidden_size, moe_intermediate_size]`.

Router math already matched (`Qwen3MoeTopKRouter.forward`: softmax over the
full row, then top-k, then optional renorm — exactly `moe_router_top_k`'s
contract), and field names from M4 (`num_experts`, `num_experts_per_tok`,
`norm_topk_prob`, `moe_intermediate_size`, `decoder_sparse_step`,
`mlp_only_layers`, `router_aux_loss_coef`) were also all confirmed correct.
Only the per-expert weight *storage shape* was wrong.

**This was surfaced to the user before any M6 code was written**, since it
meant revising already-committed M5 work, not just adding new M6 code. Given
two options — (a) change M5's internal representation to match the real
checkpoint, keeping `ops::moe_expert_ffn`'s existing separate-gate/up interface
and doing the format conversion at forward time, or (b) change
`ops::moe_expert_ffn` itself to accept HF's fused layout directly, reopening
M1–M3 — the user chose (a).

## What changed

- `MoeFeedForward` (`src/model/model.cpp`) now stores `gate_up_proj_` and
  `down_proj_` as plain `Value`s in HF's *exact* native layout (not this
  repo's usual `[input,output]` Linear convention), so `qwen_style_weight_mapping`
  maps both with a plain `Identity` transform — no per-expert split or
  transpose at load time. `router_` stays a `Linear` (unchanged from M5); its
  `[num_experts,dim]` HF layout already gets `Transpose2D` like every other
  Linear in this mapping.
- New checkpoint-format adapter, `ops::moe_split_gate_up`/
  `moe_split_gate_up_backward` (CPU only) plus `autograd::moe_split_gate_up`:
  splits `gate_up_proj` into the two `[num_experts,dim,ffn_dim]` tensors
  `moe_expert_ffn` expects, transposing each half from HF's `[out,in]` to this
  repo's `[in,out]` convention in the same step. This is explicitly a
  format-adapter op, not a new routing primitive — `down_proj` needs only a
  plain `autograd::transpose(down_proj_, 1, 2)` (already existed, no new op).
  `moe_split_gate_up`'s backward is a genuine two-parent scatter: gate's and
  up's backward closures each fill in only their own half of the reconstructed
  gradient (zero elsewhere, no overlap) and rely on `accumulate()`'s existing
  summing behavior to combine both contributions into `gate_up_proj`'s full
  gradient, rather than writing a joint two-input backward closure.
- `MoeFeedForward::forward()`/`forward_tensor()` now actually compute:
  router logits (`Linear::forward`/`forward_tensor`) → `moe_router_top_k` →
  `moe_split_gate_up` → `transpose` (down) → `moe_expert_ffn` → `moe_combine` —
  the same sequence for both the autograd (`Value`) and graph-free (`Tensor`)
  paths. `forward_normalized_bf16_tensor` now just calls
  `norm.forward_tensor(input)` then `forward_tensor` (no BF16 fusion — that
  stays unimplemented, see below).
- `qwen_style_weight_mapping`'s MoE branch rewritten to match: one
  `Transpose2D` entry for the router, two `Identity` entries
  (`moe.gate_up_proj`, `moe.down_proj`) replacing the old per-expert loop.

## What still throws

BF16/FP8/INT8 preparation and the BF16 FFN Arena remain unimplemented for MoE
(`append_bf16_training_mirrors`, `append_fp8_inference_linears`,
`append_int8_inference_linears`, `prepare_bf16_prefill_up_candidate`,
`up_weight_data`). `MoeFeedForward::require_fp32_moe()` also rejects forward
itself if `config.linear_precision != Float32` — there is no BF16/FP8 MoE
expert path to fall back to, so running forward under a non-FP32 precision
policy would otherwise silently compute in FP32 while claiming a different
precision was honored.

## Verification

Three layers:

1. **CPU hand/finite-difference tests** for the new adapter op
   (`tests/ops/ops_test.cpp`: `MoeSplitGateUpMatchesHandValuesAndRoundTripsThroughBackward`
   — the backward is the exact inverse scatter of the forward gather, so
   feeding the forward's own outputs back in must reconstruct the input, which
   it does; `tests/autograd/autograd_test.cpp`:
   `MoeSplitGateUpBackwardMatchesFiniteDifference`).
2. **Model-level tests** (`tests/model/weights_test.cpp`, rewritten for the
   new tensor layout): `MoeStateDictHasExactTensorCountAndMatchesMapping` (a
   layer now contributes exactly 3 MoE tensors regardless of expert count —
   `9` per layer total, not `9 + 3×experts`), the strict-mode
   missing/unexpected/incompatible triple adapted to the new tensor names, and
   — new — `MoeForwardProducesFiniteLogitsAndTrainsEveryParameter`: builds a
   1-layer, 2-expert model, runs `forward()`/`forward_inference()`, checks
   finite logits, then runs `loss().backward()` and confirms **every** named
   parameter (including attention and dense-unrelated ones) receives a
   gradient. `MoeAdvancedInferencePreparationIsStillExplicitlyUnimplemented`
   confirms the BF16/FP8 preparation paths still throw.
3. **The actual model-level full-graph gate**: `tests/torch/operator_oracle.cpp`
   gained `emit_moe_model_gate_case()`, reproducing `MoeFeedForward::forward()`'s
   exact call sequence (matmul → `moe_router_top_k` → `moe_split_gate_up` →
   `transpose` → `moe_expert_ffn` → `moe_combine`) with a deterministic fixture
   (3 tokens, dim 4, 4 experts, k=2, ffn_dim 3 — a "shrunk fixture" per the
   plan, not the real 128-expert config) generated by pure integer arithmetic
   on the flat index (`((i*37+11+offset*97) % 23 - 11) * 0.05`) so both
   languages produce bit-identical fixture values without sharing an RNG.
   `python/tests/test_operator_parity.py` builds the real
   `transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock`
   (transformers 5.8.0, via the conda `torch` env — the same environment used
   to close the M1 verification gap), copies the identical fixture weights
   into it (`gate.weight` transposed to HF's `[num_experts,dim]` layout;
   `gate_up_proj`/`down_proj` copied as-is since internal storage now matches),
   runs its real, unmodified `forward()` plus backward, and compares output +
   every gradient (input, router weight, gate_up_proj, down_proj) at the
   `moe_model_*` tolerance (`2e-3`, matching the plan's model-level threshold).
   Attention/embedding are deliberately excluded from this comparison — they
   are unchanged by this milestone and have their own pre-existing oracle
   (`emit_model_graph_case`, a hand-rolled equivalent); conflating the two
   would make a MoE-specific gate fail on a pre-existing, unrelated
   RoPE/attention convention question rather than on anything this milestone
   changed.

```text
CPU Debug (MICROLLM_ENABLE_HIP=OFF): 301/301 microllm_tests passed (was 298
after M5; net +3: two new adapter-op tests, and the three old M5 weight tests
became four M6 tests after the layout rewrite).
CPU ASan/UBSan (MICROLLM_ENABLE_SANITIZERS=ON): 301/301 passed clean.
scripts/audit_test_coverage.py: pass tensor_ops=207 graph_api=49 test_files=159
TorchOps.OperatorParity (conda torch env, transformers 5.8.0):
  test_every_declared_invalid_shape_or_dtype_is_rejected ... ok
  test_every_numeric_case_has_a_pytorch_reference_and_matching_shape ... ok
  test_forward_and_backward_values_match_pytorch ... ok   <- includes the real
                                                              Qwen3MoeSparseMoeBlock
                                                              comparison
  test_model_graph_snapshot_has_real_topology ... ok
```

## Current boundary

Attention/embedding paths are unchanged and untested by this milestone's new
oracle (they're covered by the pre-existing dense-model gate). No BF16/FP8/INT8
MoE support. No HIP: `ops::moe_split_gate_up`/`_backward` are CPU-only, and
`MoeFeedForward`'s forward calls into `autograd::moe_expert_ffn` etc., whose
HIP kernels (M2) remain unverified on real hardware — running this model's
`forward()` on a HIP device would exercise that still-uncompiled code path for
the first time. No CLI/serving integration, no real-checkpoint end-to-end test
(that's M7, and it can now proceed with confidence in the field names and
weight layout, both checked against transformers 5.8.0's actual source in this
milestone and M4).
