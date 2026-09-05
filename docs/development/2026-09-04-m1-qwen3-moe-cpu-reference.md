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

The three-way CPU/PyTorch oracle (`tests/torch/operator_oracle.cpp` emit cases plus
matching `python/tests/test_operator_parity.py` `record()` calls and
`invalid_moe_*` rejection names) was run against a CUDA-built PyTorch from a local
conda env (`conda activate torch`; `torch==2.11.0+cu130`) by pointing
`CMAKE_PREFIX_PATH` at its `torch/share/cmake` directory. `find_package(Torch)`
succeeds against a CUDA build without a working CUDA runtime being present, and
`microllm_operator_oracle` itself has no torch dependency at all (it only links
`microllm::model`/`microllm::training`), so this validates real values, not just
configuration:

```text
$ MICROLLM_OPERATOR_ORACLE=.../microllm_operator_oracle python python/tests/test_operator_parity.py -v
test_every_declared_invalid_shape_or_dtype_is_rejected ... ok
test_every_numeric_case_has_a_pytorch_reference_and_matching_shape ... ok
test_forward_and_backward_values_match_pytorch ... ok
test_model_graph_snapshot_has_real_topology ... ok
Ran 4 tests in 1.349s — OK
```

`test_forward_and_backward_values_match_pytorch` is the one that matters for this
milestone: it confirms `moe_router_top_k`'s softmax+top-k+renorm, `moe_expert_ffn`'s
masked all-experts SwiGLU, and `moe_combine`'s weighted sum are bit-for-bit
consistent (within the default `3e-5` tolerance) with independently written PyTorch
formulas, not just internally self-consistent. This closes the verification gap
this record originally carried — the earlier draft of this paragraph said the
parity test could not be run in this environment; it has now been run.

## Current boundary

No `.hip` kernel, no autograd node, no config parsing, and no weight loading exist
for these three ops yet. The next milestone (M2) implements the readable HIP kernels
and the three-way CPU/HIP/PyTorch oracle comparison in `tests/ops/hip_ops_test.cpp`.
