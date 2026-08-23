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
- optimized implementation policy: `src/ops/optimized.cpp`.

## Candidate selection today

The current registry covers exact 2D matmul shapes and selects between readable HIP and
hipBLASLt. It is a tested seam, not a mature general autotuner. A production registry
still needs architecture, dtype, layout, mode, workspace, runtime/library version,
candidate metadata, persistent cache, and invalidation keys.

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
