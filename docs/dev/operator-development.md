# Add or optimize an operator

## Correctness-first workflow

1. Write dtype, device, shape, layout, output, and error contracts.
2. Add hand-valued CPU reference tests.
3. Add legal and illegal shape cases.
4. Add an independent PyTorch forward oracle.
5. If differentiable, add PyTorch autograd and finite-difference gates.
6. Implement the readable HIP kernel.
7. Compare CPU, PyTorch, and HIP.
8. Integrate the operator into the graph/model.
9. Profile the end-to-end workload.
10. Add optimized candidates only for measured hotspots.

## Public interfaces

- owning convenience operations: `include/microllm/ops/ops.h`;
- non-owning output/view boundary: `include/microllm/ops/low_level.h`;
- Stream/workspace selection: `include/microllm/ops/context.h`;

Arena-backed owning-style operators may also wrap caller memory with
`Storage::from_external`, then construct a Tensor using `Tensor::from_storage`. The wrapper does
not extend lifetime or free the pointer. Prefer `_out_` operators such as `matmul_out_` and
`swiglu_out_`; never return an arena-backed Tensor beyond its arena/Stream lifetime.
For BF16 FFN, use `Bf16FfnWorkspace` and `bf16_ffn_out_`. The fallback is mandatory even when the
development GPU accepts direct FP32 output: support is exact-shape/runtime dependent, and a caller
must not discover an allocation or unsupported error only after Graph capture.
At model level, never create one workspace per block. The opt-in cache owns one backing per exact
row count and shares it across sequential blocks. Returned workspace views must not escape the
model forward, and concurrent calls require external synchronization.
Selection must use shape/runtime facts rather than model names. When `minimum_rows` bypasses a
call, no backing Storage may be created and allocation/peak counters must match the old path.
`Bf16QkvWorkspace` follows the same rule but has three independent fallback/output widths. Never
reuse one fallback for K and V merely because one model happens to give them equal dimensions;
the public contract rejects all writable aliases.
When proposing another persistent workspace, first add a fixed `AllocationSource` scope and inspect
exact sizes. Dynamic diagnostic strings are forbidden because the profiler would allocate while
trying to measure allocation. Disabled scopes must remain one-branch no-ops.
For long causal GQA, expanded K and expanded V may share Storage only because QK is submitted before
the V repeat on the same Stream. Different Streams or reordering invalidate that liveness proof.
For vendor solution indices, retain the full descriptor key and backend version. Never paste an
index from a different ROCm build or register a candidate that was timed before complete-output
finite/Max/RMS checks.
For FP32 batched GEMM, use `make_fp32_matmul_solution_key` instead of constructing a partial key.
The key flattens leading batch dimensions exactly as hipBLASLt does and includes alpha/workspace/
environment identity. `register_fp32_matmul_solution` is thread-local and explicit. Inspect
`fp32_matmul_solution_stats()` to prove a candidate actually dispatched; zero hits are not a
performance result. Even a bit-exact operator candidate still needs complete-model throughput and
peak-memory gates before any default can change.
BF16 GroupedQKV has an even stricter lifetime rule: `GroupedGemm::initialize` binds every pointer.
Use it only through `bf16_qkv_projection_out_` with caller-owned stable buffers. The exact cache key
must include all three weights, BF16 intermediates, FP32 outputs, device and Stream. Timing only
`run()` is valid for a cache hit; a separate reinitialized measurement must prove why cacheability
is required. Direct grouped FP32-output rejection must remain visible rather than silently changing
the precision boundary.
When many pointer sets share one grouped shape, do not initialize one GroupedGemm per pointer set.
Initialize one kernel with device user arguments, cache the algorithm/kernel by exact environment,
and store only a device argument record per pointer plan. Report kernel setup and argument setup
separately; warm steady-state timing cannot justify hiding first-use latency.
Grouped gate/up follows the same rule. Experiment 195 shows the stable two-operation path is faster
while per-call initialization is slower on both official shapes. A production implementation must
bind the existing FFN Arena input/gate/up addresses and each block's persistent weights; temporary
outputs or a stateless convenience wrapper do not satisfy the evidence contract.
- readable HIP launch declarations: `src/ops/hip/kernels.h`;
- optimized matmul policy: `src/ops/optimized.cpp`;
- correctness-first matmul/AdamW tuners: `src/ops/tuning.cpp` and
  `src/ops/adamw_tuning.cpp`.

## Candidate selection today

The current registries cover exact 2D implementation choices, exact batched FP32 hipBLASLt
solution indices and flat AdamW state updates.
Matmul isolates dtype, layout/strides, mode, workspace and backend versions; AdamW isolates
element count, mirror presence, every state pointer's alignment, mode and HIP environment.
Both have transactional persistent caches and reject stale environments. Their tuners
compare complete output/state before timing and never register a winner implicitly.

This is a measured offline selection seam, not permission to generalize one result to all
shapes or models. Each additional operator still needs its own exact key, supported-domain
checks, correctness evidence and end-to-end acceptance.

## Acceptance rule

An optimized candidate is accepted only if it:

- passes the same shape/error and numerical tests as the reference;
- improves repeated device time for its declared domain;
- does not regress the target end-to-end workload beyond the documented budget;
- falls back safely outside its domain;
- leaves the readable implementation available for diagnosis.

For FP8 Linear weights with different output-channel ranges, see the
[output-column scale operator record](../development/2026-08-23-fp8-output-column-scale-operator.md).
It explains why the MI300 path uses native scalar-scale GEMM followed by a device column scale,
instead of silently falling back when outer-vector scale is unavailable.

For outlier clipping in dynamic FP8 Tensor scales, read the
[clipped dynamic quantization record](../development/2026-08-23-fp8-clipped-dynamic-quantization.md).
The default fraction is compatibility-preserving; model policy requires separate evidence.

For mixed FP8 formats, see the
[E5M2 activation/E4M3 weight probe](../development/2026-08-23-fp8-mixed-e5-activation-probe.md).
Header availability is not accepted; the probe records native dispatch and fallback counts.
