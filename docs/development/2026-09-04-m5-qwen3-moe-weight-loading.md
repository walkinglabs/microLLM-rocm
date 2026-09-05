# 2026-09-04 — M5 Qwen3 MoE weight loading

## Contract

Give `TransformerModel` real per-expert MoE parameters so the existing strict/
non-strict `load_state_dict`/`state_dict` machinery works end to end for MoE
configs — missing/unexpected/incompatible all rejected per `docs/WEIGHTS.md`,
plus an exact total-tensor-count assertion that would catch a silently-dropped
expert. Forward and every one-way inference preparation stay explicitly
unimplemented; that's M6.

## Scope decision, made with the user

`TransformerModel`/`Block`/`FeedForward` is a single 4167-line file deeply
integrated with BF16/FP8/INT8 precision and the BF16 FFN Arena, with 18 call
sites inside `Block` touching its dense FFN member. The user was asked whether
M5 should (a) fully integrate MoE parameters into this file so the real strict
loader exercises them, or (b) build a decoupled weight-loading module now and
integrate later at M6. They chose (a) — full integration, matching the M5 plan's
literal wording ("strict-mode triple... rejected per docs/WEIGHTS.md" only means
something against a real loader). This record is written assuming that choice;
the tradeoff accepted is a larger, riskier diff through a file with no way to
verify the advanced features on real hardware from this machine.

## What was added

- `MoeFeedForward` (`src/model/model.cpp`, right after `FeedForward`): one
  router `Linear` (`dim -> num_experts`, no bias) plus three
  `std::vector<Linear>` of size `num_experts` for gate/up/down projections —
  real per-expert `Linear` objects, each already carrying the model's
  `linear_precision`/FP8 settings from `Linear`'s own constructor, reused as-is
  rather than duplicating that logic.
- `Block` now holds `std::optional<FeedForward> feed_forward_` and
  `std::optional<MoeFeedForward> moe_feed_forward_` plus an `is_moe_` flag set
  from `config.moe_num_experts > 0`; exactly one is populated. Every one of the
  18 former `feed_forward_.xxx()` call sites became an `is_moe_ ? ... : ...`
  branch.
- `append_named` on `MoeFeedForward` registers `<prefix>.router.weight` and, per
  expert, `<prefix>.experts.<E>.{gate_proj,up_proj,down_proj}.weight` — matching
  the M0 plan's literal naming example.
- `qwen_style_weight_mapping` gained a MoE branch mapping each of those internal
  names to an assumed external Hugging Face name
  (`model.layers.N.mlp.gate.weight` for the router,
  `model.layers.N.mlp.experts.E.*_proj.weight` per expert), transposed like
  every other Linear weight in this mapping.
- Two narrow correctness fixes elsewhere, both needed for the loader to behave
  and neither touching the dense path:
  - `TransformerModel`'s constructor cross-checks
    `parameter_count() == config.parameter_count()`; `ModelConfig::parameter_count()`
    already throws for MoE (M4), so that check is now skipped when
    `moe_num_experts != 0` instead of turning every MoE model construction into
    an unrelated exception.
  - `prepare_bf16_ffn_inference(fp32_layers, scope)` selects weights to convert
    by a pure name filter (`.feed_forward.` + a `*_proj.weight` suffix) over
    every `named_parameters()`, not by calling any `Block`/`FeedForward` method.
    MoE names use `.moe.` instead of `.feed_forward.`, so the filter already
    matched zero of them and the existing `converted_tensors != expected_count`
    check already threw — but only by accident, with a confusing message. Added
    an explicit `moe_num_experts != 0` guard at the top for a clear error
    instead of relying on that coincidence.

## What deliberately still throws

`MoeFeedForward::forward`/`forward_tensor`/`forward_normalized_bf16_tensor` throw
`std::logic_error` unconditionally — no forward integration this milestone. So do
`prepare_bf16_prefill_up_candidate`, `up_weight_data` (there is no single "the up
weight" for MoE — each expert has its own), `append_bf16_training_mirrors`, and
`append_fp8_inference_linears`/`append_int8_inference_linears`. Two methods are
no-ops instead of throws, deliberately: `set_bf16_ffn_arena_cache` (disabling an
opt-in arena is always safe) and `commit_bf16_prefill_up_candidate` (its `Block`
wrapper is `noexcept`, and it is unreachable in practice because
`prepare_bf16_ffn_decode_up_fp32_inference()` already calls
`prepare_bf16_prefill_up_candidate()` — which throws — on every block before any
commit is attempted; a no-op here can't mask that path since nothing takes it).
`Block::ffn_up_weight_data()` had its `noexcept` removed — it's an internal
implementation class local to `model.cpp`, never declared in `model.h`, so this
carries no external compatibility concern — since there is no safe non-throwing
value it could return for MoE.

Verified with a dedicated test
(`ModelWeightsTest.MoeForwardAndAdvancedInferencePreparationAreExplicitlyUnimplemented`)
that `forward`, `forward_inference`, `prepare_bf16_ffn_inference`, and (on a
separate FP8-configured model) `prepare_fp8_inference_weights` all throw for an
MoE model rather than silently running.

## An assumption worth flagging, same caveat as M4

The external Hugging Face naming used in `qwen_style_weight_mapping`'s new MoE
branch (`mlp.gate.weight` for the router, `mlp.experts.E.{gate_proj,up_proj,
down_proj}.weight` per expert) follows the Qwen2Moe/Mixtral convention and has
not been checked against a real Qwen3-MoE `config.json`/checkpoint — this
machine still has no network access. Re-verify against a real checkpoint at M7.

## Verification

```text
CPU Debug (MICROLLM_ENABLE_HIP=OFF): 298/298 microllm_tests passed (was 295
after M4; +3 new MoE weight-loading tests).
CPU ASan/UBSan (MICROLLM_ENABLE_SANITIZERS=ON): 298/298 passed clean -- no
memory or UB issues from the std::optional<FeedForward>/std::optional<
MoeFeedForward> refactor or the new Linear-vector-backed MoeFeedForward.
scripts/audit_test_coverage.py: pass tensor_ops=205 graph_api=48 test_files=159
(unaffected; model.h is not part of its audited public-symbol surface).
```

New tests (`tests/model/weights_test.cpp`):
- `MoeStateDictHasExactPerExpertTensorCountAndMatchesMapping` — a 2-layer,
  3-expert model's `named_parameters()`/`state_dict()`/`qwen_style_weight_mapping()`
  sizes all equal a hand-computed closed-form count; a dropped expert would show
  as an off-by-3 mismatch, not a vague size difference.
- `MoeStrictLoadRejectsMissingUnexpectedAndIncompatibleExpertTensors` — the
  actual strict-mode triple, mirroring the existing dense
  `StrictLoadIsAtomicAndNonStrictLoadReportsEveryProblem`: a deleted expert
  tensor (missing), a wrong-shaped one (incompatible), and an extra one
  (unexpected) are all reported, strict mode throws and leaves the target
  unmodified, non-strict reports exactly one of each.
- `MoeForwardAndAdvancedInferencePreparationAreExplicitlyUnimplemented` — see above.

No regression to any existing dense-model test; the `is_moe_` branches are
inert (`false`) for every config that predates this change.

## Current boundary

No forward integration, no autograd wiring from `TransformerModel::forward()`
into `autograd::moe_router_top_k`/`moe_expert_ffn`/`moe_combine` (those exist
from M3 but are not yet called from the model). No BF16/FP8/INT8 support for MoE
experts, no BF16 FFN Arena support for MoE. All of that is M6 ("model-level
full-graph gate"), which also needs the actual HF Transformers Qwen3MoE forward
as its oracle — not available to write against without confirming the M4/M5
field-name assumptions first.
