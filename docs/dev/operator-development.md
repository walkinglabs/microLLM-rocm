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
- readable HIP launch declarations: `src/ops/hip/kernels.h`;
- optimized matmul policy: `src/ops/optimized.cpp`;
- correctness-first matmul/AdamW tuners: `src/ops/tuning.cpp` and
  `src/ops/adamw_tuning.cpp`.

## Candidate selection today

The current registries cover exact 2D matmul problems and flat AdamW state updates.
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
