# 2026-09-04 — M0 Qwen3 MoE operator contracts

## Contract

This is the paper-only milestone from the Qwen3 MoE plan: define the public shape,
dtype, and PyTorch-oracle contract for three new operators before touching `ops.h`,
`coverage_manifest.json`, or any `.cpp`/`.hip` file. No code lands in this change.
The full contract text lives in
[`docs/OPERATOR_CONTRACTS.zh-CN.md`](../OPERATOR_CONTRACTS.zh-CN.md), under the "MoE
路由" heading; this record captures the decision context and the next-milestone
boundary. (As of M1, that heading and its implementation status have moved on from
this record's "M0, not implemented" framing — see
[the M1 CPU reference record](2026-09-04-m1-qwen3-moe-cpu-reference.md).)

## Observed state before this change

`grep -r moe src/ include/` finds nothing. The repository supports Qwen2 and Qwen3
as dense-only architectures; there is no router, no top-k operator, and no per-expert
weight loading path anywhere in the tree.

## Three operators proposed

- `moe_router_top_k`: router logits `[tokens, num_experts]` to `(indices[tokens,k]
  Int32, weights[tokens,k] FP32)`. Oracle is softmax over the full expert row, then
  `torch.topk`, with an optional renormalization over just the k selected weights
  (`norm_topk_prob`), matching the Qwen3 MoE routing block.
- `moe_expert_ffn`: deliberately naive first version. Computes the SwiGLU FFN for
  every expert against every token (`O(num_experts)`, not `O(k)`), then masks out
  experts that were not selected for a given token. No gather or dispatch yet — that
  is explicitly deferred to a later performance milestone.
- `moe_combine`: weighted sum of the k selected expert outputs back down to
  `[tokens, dim]`, using the router weights from `moe_router_top_k`.

## Decisions

- Contracts are written before any implementation, per the correctness-first
  operator workflow (`docs/dev/operator-development.md` step 1).
- The naive compute-all-experts-then-mask shape is chosen over sparse dispatch for
  the first version so correctness can be validated independently of a gather
  kernel. Grouped/sparse expert dispatch is explicitly deferred to the performance
  goal, not silently assumed here.
- Backward is out of scope for M0/M1/M2; the contract only records that top-k
  selection itself is not differentiable and that gradient must flow like
  `embedding_backward`'s scatter-add — only visited `(token, expert)` rows receive
  gradient, everything else must be exactly zero. This becomes binding once M3
  (autograd backward nodes) starts.
- Every new op name will need a matching entry in `tests/coverage_manifest.json` the
  moment `ops.h` gains the declaration; that sync is scheduled for M1, not here.

## Current boundary

This change is contracts only. It does not add CPU reference code, HIP kernels,
config parsing, weight loading, or coverage manifest entries. The next milestone
(M1) implements the naive CPU float32 reference for these three operators, gated by
a PyTorch oracle (`torch.topk` + masked-softmax-renorm + manual masked SwiGLU loop)
with exact-value comparison, and registers all three names in
`tests/coverage_manifest.json` so `scripts/audit_test_coverage.py` does not fail CI.
