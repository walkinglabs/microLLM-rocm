# 2026-09-04 — M3 Qwen3 MoE autograd backward nodes

## Contract

Add CPU backward primitives and `autograd::Value`-level wiring for the three MoE
operators, gated by PyTorch autograd over the forward — not a hand-written formula
— per this repo's standing backward-primitive rule.

## What's differentiable and what isn't

Top-k selection is not differentiable: `expert_indices` is always a plain `Tensor`
input/output, never a `Value`, matching `embedding()`'s existing convention for its
`indices` argument. Three raw backward primitives were added
(`moe_router_top_k_backward`, `moe_expert_ffn_backward`, `moe_combine_backward`),
each recomputing the necessary forward intermediates internally rather than caching
them, matching the convention already used by e.g. `rms_norm_backward`.

- **`moe_router_top_k_backward`** reuses the existing `softmax`/`softmax_backward`
  primitives rather than re-deriving the Jacobian: it recomputes the full-row
  softmax, applies the gather/optional-renormalize backward analytically to get a
  gradient in softmax-output space, scatters that into the full `[tokens,
  num_experts]` shape (zero at non-selected slots), then calls
  `softmax_backward`. One consequence worth flagging because it's easy to
  misremember: the *router logit* gradient is **dense**, not sparse — softmax
  couples every logit through its shared denominator, so even experts nobody
  selected still receive a nonzero logit gradient. This is mathematically correct
  and matches PyTorch exactly; only the *selection itself* is non-differentiable,
  not the softmax that produced it.
- **`moe_expert_ffn_backward`** returns `TensorQuad{input, gate_weight, up_weight,
  down_weight}` (a new `TensorQuad` struct was added next to `TensorPair`/
  `TensorTriple`). It reuses `ops::swiglu`/`ops::swiglu_backward` for the
  nonlinear step instead of re-deriving the SiLU derivative. Because the forward
  masks each `(token, expert)` output to exactly zero when the token didn't select
  that expert, the *local* Jacobian of that output w.r.t. that expert's weights is
  also exactly zero there — so an expert's weight-row gradient only accumulates
  contributions from tokens that actually selected it. This is the same "only
  visited rows get gradient" property `embedding_backward` has, but here it falls
  out of the mask-multiply automatically; no explicit scatter bookkeeping was
  needed.
- **`moe_combine_backward`** returns `TensorPair{expert_output, expert_weights}`.
  `expert_output`'s gradient genuinely is a scatter-add (only the `(token,
  expert)` slots `moe_combine`'s forward actually read receive a nonzero
  contribution) — this is the one case in this milestone that's a literal,
  hand-written scatter rather than a mask-multiply side effect.

## Autograd wiring

`autograd::moe_router_top_k` returns a new `MoeRouterResult{Tensor indices; Value
weights;}` — the mixed differentiable/non-differentiable return shape needed a new
struct since `ValueTriple` only holds `Value`s. `autograd::moe_expert_ffn` and
`autograd::moe_combine` follow the existing `operation(name, output, {parent
nodes}, backward_lambda)` pattern used throughout `autograd.cpp` (e.g.
`cross_entropy`), each calling the matching raw backward primitive and routing the
resulting gradients to their respective parent nodes via `accumulate`.

## Verification

Three layers, all passing:

1. **CPU hand/finite-difference tests** (`tests/ops/ops_test.cpp`): the router and
   expert-FFN backward primitives are checked against numerical (central
   finite-difference) gradients rather than hand-derived closed forms, since
   hand-computing the router's dense-softmax gradient by hand across enough
   decimal places to be trustworthy is error-prone; combine's backward is linear
   and checked against exact hand values instead. The "non-selected expert gets
   exactly zero" property is checked with `EXPECT_EQ`, not `EXPECT_NEAR`.
2. **Graph-level CPU test** (`tests/autograd/autograd_test.cpp`,
   `MoeRouterExpertFfnAndCombineBackwardsRouteThroughTheFullGraph`): chains
   `moe_router_top_k` → `moe_expert_ffn` → `moe_combine` through one `Value` graph
   and one `.backward()` call, then checks the same "unvisited expert gets exact
   zero gradient" property survives the full `operation()`/`accumulate()` routing,
   not just the isolated raw-op unit test.
3. **Real PyTorch parity** (`tests/torch/operator_oracle.cpp` +
   `python/tests/test_operator_parity.py`, `graph_moe_*` cases): built
   `microllm_operator_oracle` against a CUDA-built PyTorch from a local conda env
   (`conda activate torch`, torch==2.11.0+cu130) by pointing `CMAKE_PREFIX_PATH`
   at its `torch/share/cmake` — same technique used to close the M1 verification
   gap. All four `TorchOps.*` tests pass, including
   `test_forward_and_backward_values_match_pytorch`, which is the actual "backward
   reference = PyTorch autograd over the forward" standard this milestone commits
   to, not just an internal-consistency check.

```text
CPU Debug (MICROLLM_ENABLE_HIP=OFF): 292/292 microllm_tests passed (was 291 after
M2; +1 for the new graph-level MoE test).
scripts/audit_test_coverage.py: pass tensor_ops=205 (was 202) graph_api=48 (was 45)
TorchOps.OperatorParity (conda torch env, CUDA build, CPU-only oracle binary):
  test_every_declared_invalid_shape_or_dtype_is_rejected ... ok
  test_every_numeric_case_has_a_pytorch_reference_and_matching_shape ... ok
  test_forward_and_backward_values_match_pytorch ... ok
  test_model_graph_snapshot_has_real_topology ... ok
```

## Current boundary

CPU only. `moe_router_top_k_backward`/`moe_expert_ffn_backward`/
`moe_combine_backward` all throw explicitly on a HIP Tensor
(`"... has no HIP kernel yet; CPU reference only"`), matching M1's staging before
M2 added forward HIP kernels. Backward HIP kernels were deliberately not attempted
here: M2's forward HIP kernels are themselves still unverified on real hardware
(no ROCm toolchain on this machine — see the M2 record), and writing a second layer
of uncompiled HIP code on top of an already-uncompiled layer would only compound
that risk without a way to catch mistakes. HIP backward is deferred to whenever M2
gets its first real hardware verification pass.

No config parsing, no weight loading, no model-level integration. The next
milestone (M4) is config parsing (`num_experts`, `num_experts_per_tok`,
`norm_topk_prob`, `moe_intermediate_size` in `model/huggingface.cpp`).
