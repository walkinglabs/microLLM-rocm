# 2026-09-04 — M2 Qwen3 MoE readable HIP kernels

## Contract

Add readable HIP kernels for the three MoE operators contracted in M0 and given a
CPU reference in M1, mirroring the CPU math exactly (same top-k tie-break order,
same compute-all-experts-then-mask FFN cost model). Wire real dispatch into
`ops.cpp`'s `device().is_hip()` branches, replacing the M1 placeholder throws.

## Important limitation: unverified on real hardware

This development machine has no ROCm/HIP toolchain and no AMD GPU (`hipcc`,
`rocminfo`, `/opt/rocm*` all absent). The kernel code in
`src/ops/hip/basic_kernels.hip` and the declarations in `src/ops/hip/kernels.h`
were written and reviewed by hand against the repo's existing kernel idioms, but
**have never been compiled by hipcc or executed on a device.** Only the CPU-facing
half of this change is compiler-verified: the C++ dispatch added to `ops.cpp`
(Tensor allocation, `require_contiguous` calls, the `hip::launch_moe_*` call sites)
sits outside the `#if MICROLLM_HAS_HIP` guard except for the actual kernel-launch
calls, so a `MICROLLM_ENABLE_HIP=OFF` build exercises all of it except the launches
themselves — confirmed by rebuilding `microllm_tests` and rerunning the CpuOpsTest
suite (55/55 pass, unchanged from M1) after this change.

The three-way CPU/HIP/PyTorch oracle gate this milestone's test standard calls for
(`tests/ops/hip_ops_test.cpp`, threshold aligned to the existing softmax-class ops
at `2e-6,2e-5`) has **not been written or run**, per explicit agreement with the
user to defer ROCm-side testing until AMD hardware is available. Treat the kernel
code below as reviewed-but-unbuilt until that happens.

## Kernel design

All three kernels are "readable" tier: one thread per output row/element, no
shared-memory reduction, no vectorization — matching the plan's explicit call for
"simple per-token full-comparison top-k... no need for fancy selection network
yet."

- `moe_router_top_k_kernel`: one thread per token. Selects k experts by repeatedly
  scanning the full row for the softmax value ranked immediately after the
  previous pick, using only two scalar registers (`previous_value`,
  `previous_index`) as selection state — no per-token array sized by
  `num_experts` is ever materialized, so there is no artificial expert-count cap.
  The comparator (`value > best || (value == best && index < best_index)`) matches
  the tie-break idiom already used by `argmax_kernel`/`argmax_last_dim_kernel`
  elsewhere in this file, which is what makes it safe to assume it reproduces
  `torch.topk`'s stable CPU tie order the same way the CPU reference's
  `std::stable_sort` does.
- `moe_expert_hidden_kernel` + `moe_expert_down_kernel`: `moe_expert_ffn` is split
  into two kernel launches through a caller-allocated `[tokens, num_experts,
  ffn_dim]` FP32 device workspace, because a single-kernel-per-(token,expert)
  design would need either a per-thread `ffn_dim`-sized hidden array (same
  unbounded-local-storage problem the router avoids) or redundant recomputation
  that blows up the complexity class. The down kernel computes every expert's
  projection unconditionally and multiplies by a 0/1 mask at the very end —
  mirroring the CPU reference's mask-multiply structure exactly, so the O(num_experts)
  cost contract holds on the down projection too, not just the hidden projection.
- `moe_combine_kernel`: one thread per (token, dim) output element, reading only
  the k selected experts. Guards against an out-of-range expert index by writing
  NaN rather than performing an out-of-bounds device read — the same defensive
  pattern `embedding_kernel` already uses for out-of-range vocabulary indices,
  chosen over host-side value validation because the existing HIP tier never
  round-trips device tensor contents to host for validation (see `embedding`'s HIP
  path, which has no analogous CPU-side range check either).

## Decisions

- No artificial `num_experts` upper bound was introduced. The router's O(1)-state
  selection algorithm (see above) made this unnecessary rather than requiring a
  documented cap.
- `moe_expert_ffn`'s HIP path always allocates a full `[tokens, num_experts,
  ffn_dim]` workspace Tensor per call. This is the naive/dense M1-M2 baseline;
  a workspace-reuse or arena-backed version is out of scope until performance work
  begins on top of a correctness baseline, matching this repo's standing rule that
  optimized candidates need measured evidence, not anticipation.
- HIP-side illegal-input behavior intentionally diverges from the CPU reference in
  one place: `moe_combine`'s out-of-range expert index produces NaN on HIP instead
  of `std::out_of_range` on CPU. This matches the existing `embedding` operator's
  precedent (NaN sentinel on HIP vs. a CPU throw) rather than introducing a new
  host-validation pattern nothing else in the file uses.

## Current boundary

No autograd node, no config parsing, no weight loading. Before this milestone can
be considered actually done (not just written), it needs, on a ROCm machine:

1. `cmake -DMICROLLM_ENABLE_HIP=ON` actually compiling `basic_kernels.hip` with
   these additions — the very first check that hasn't happened yet;
2. the three-way CPU/HIP/PyTorch oracle in `tests/ops/hip_ops_test.cpp`;
3. non-contiguous-HIP-input rejection tests (`require_contiguous` calls exist in
   the `ops.cpp` dispatch but are untested here).

The next milestone (M3, once M2 is hardware-verified) is autograd backward nodes.
