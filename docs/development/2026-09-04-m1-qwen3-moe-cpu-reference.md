# 2026-09-04 — M1 Qwen3 MoE CPU reference operators

## Contract

Implement the naive CPU float32 reference for the three MoE operators contracted in
M0 (`docs/development/2026-09-04-m0-qwen3-moe-op-contracts.md`), gated by a PyTorch
oracle and hand-computed values, before any HIP kernel exists.

## Implemented operators

- `moe_router_top_k`: softmax over the full expert row, then top-k selection, with
  an optional renormalization over just the k selected weights;
- `moe_expert_ffn`: evaluates the SwiGLU FFN for every expert against every token
  (`O(num_experts)` per token, not `O(k)`) and masks out experts not selected for a
  given token — no gather/dispatch;
- `moe_combine`: weighted sum of the k selected expert outputs back to
  `[tokens, dim]`.

All three reject a HIP-device Tensor explicitly (`"... has no HIP kernel yet; CPU
reference only until M2"`) rather than silently falling back to host compute.

## Verification

```text
CPU Debug (MICROLLM_ENABLE_HIP=OFF): 288/288 microllm_tests passed, including the
three new CpuOpsTest.Moe* cases with hand-computed expected values (router top-2
softmax reduces to sigmoid(+-1) for the chosen two-value case; expert_ffn/combine
use small identity-matrix weights so the SwiGLU and weighted-sum arithmetic is
exact by hand).
scripts/audit_test_coverage.py: pass tensor_ops=202 (was 199) graph_api=45 test_files=159
```

The three-way CPU/PyTorch oracle wiring (`tests/torch/operator_oracle.cpp` emit
cases plus matching `python/tests/test_operator_parity.py` `record()` calls and
`invalid_moe_*` rejection names) was written and is internally self-consistent —
both sides compute the identical masked-softmax-renorm and masked-SwiGLU formula —
but this development environment has no PyTorch/libtorch install
(`bindings/torch/CMakeLists.txt`'s `find_package(Torch)` guard skips the
`microllm_operator_oracle` target entirely), so `TorchOps.OperatorParity` could not
actually be executed here. It must be run once on a machine with PyTorch before this
milestone is considered fully closed.

## Current boundary

No `.hip` kernel, no autograd node, no config parsing, and no weight loading exist
for these three ops yet. The next milestone (M2) implements the readable HIP kernels
and the three-way CPU/HIP/PyTorch oracle comparison in `tests/ops/hip_ops_test.cpp`.
