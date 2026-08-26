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
`matmul_weight_gradient_out_` is the narrower linear-backward form: it requires rank-2 input and
output gradient and writes `input^T @ output_gradient` into the validated caller Tensor. It does
not add an existing gradient; Autograd may route only a proven first/sole contribution to it.
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
Strided-copy diagnostics reuse the same source scope. The scope stays active when either allocation
or strided diagnostics is enabled, and inactive when both are off. Aggregation includes source so
identical layouts from different regions remain distinguishable.
Inference BTHD routing must reuse the existing fused RoPE and causal GQA primitives. Keep an
explicit eligibility predicate and the readable BHTD fallback; do not change cached-prefill or
trace-value layouts without separate contracts.
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
Elementwise candidates follow the same rule. `swiglu_with_implementation` and
`swiglu_out_with_implementation_` expose the measured BF16 vector route, while Auto remains scalar
because DeepSeek's full-model gate was only 1.001x. Benchmark caller-provided output Storage; timing
an owning convenience call would include allocator work and answer a different question.
The grouped gate/up Swish switch is also explicit and default-off. Its plan and kernel keys include
the epilogue bit, and the CLI rejects it without an exact grouped algorithm. `multiply_out_` keeps
the remaining gate/up product in caller Storage. Operator support is not a model claim: the MI300X
T1024 model gate regressed DeepSeek and changed complete logits, so this route must not become Auto.
For direct low-precision normalization output, compare `rms_norm_bf16_out_` against the current GPU
FP32 RMSNorm followed by the GPU cast, not against a differently ordered CPU reduction. Both paths
must use caller Storage so allocator work cannot masquerade as a Kernel result. Operator admission
does not authorize the Transformer route; the latter needs a separate complete-logit model gate.
Direct typed Softmax follows the same no-temporary rule: FP16/BF16 reductions use FP32 registers
and round only the caller output. Widths through 32 retain the readable serial row; wider rows use
64/128/256-thread block reductions. Dispatch-boundary tests must cover 32/33, 64/65 and 128/129,
while widths 2048–8192 may retain FP32 exponentials in bounded block-local LDS. Tests must also
cover 2047/2048 and 8192/8193 so an unsupported width cannot request excess shared memory.
Performance claims must keep the remaining width4096 counterexample visible. A broad wave-shuffle
reduction was removed because BF16 wall improved only 1.033× even though FP16 passed; any retry must
declare a dtype-specific predicate before measurement rather than averaging those rows. The accepted
retry uses a compile-time boolean: FP16 cached rows select wave reduction and BF16 instantiates the
same Kernel with the shared tree. Do not turn this back into runtime dtype guessing or broad promotion.
The FP16 path continues to use precise `expf`: the fast intrinsic passed the current low-precision
oracle but improved Event/wall only 1.045×/1.034×, so approximation was not admitted.
For cached FP16 only, 1024 threads is the measured MI300X winner over 128/256/512. Keep this predicate
next to the dtype/range gate; it is not permission to increase unrelated Kernel workgroups.
Attribution places PyTorch/raw/C++/Python Event at 4.530/4.764/4.815/5.086μs for `[8,4096]`.
The C++ validation layer is only 1.011× raw; optimize the adapter boundary before claiming the full
12.3% Python/PyTorch time gap is a device-Kernel problem.
That model gate now exists for FFN Norm: `bf16_ffn_precast_out_` consumes an already-filled Arena
input, Qwen/DeepSeek both pass, and enabling BF16 FFN Arena enables this exact route by default.
Keep explicit false available, and never apply the shortcut to trace, cached, training or bypassed
calls without separate evidence.
The analogous Attention path uses `bf16_qkv_projection_precast_out_`. It is default only when the
BF16 QKV Arena is enabled and hit; trace, cached, training and bypass paths retain their previous
contracts. The full gate compares against the already-retained FFN Norm default on both sides so
the two improvements are not double-counted.
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
The retained implementation dispatches only from bf16_ffn_out_ after exact registration. Tests
must prove one kernel entry, one plan miss per block, later plan hits, zero hot-path host transfer,
and the old zero-dispatch behavior when no registration exists.
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
