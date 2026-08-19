# 2026-08-19 — real FP16/BF16 Tensor storage

## Goal

Turn the existing FP16/BF16 enum and safetensors conversion into real two-byte Tensor
storage before extending math kernels.

## Implemented

- host `Float16` and `BFloat16` storage types with round-to-nearest-even conversion;
- `Tensor::from_vector(..., dtype)` for FP32/FP16/BF16;
- floating `to_vector()`, `fill()`, and explicit `cast()`;
- generic element-size-aware CPU/HIP strided materialization;
- shape/stride/view/transpose/slice/device-copy behavior independent of float width;
- preservation checks for signed zero, infinities, and NaN;
- MI300X two-byte allocation, transfer, non-contiguous materialization, and cast smoke.

## Boundary

Math operators still reject FP16/BF16 in this commit. HIP `cast()` is correctness-first
and currently stages through host memory; a device-native conversion Kernel belongs to
the operator/dtype dispatch milestone. The implementation therefore proves Tensor
storage semantics, not mixed-precision training.

## Hardware decision

The tested MI300X is CDNA3/gfx942: native FP8 is a later GEMM target, while native
MXFP4/MXFP6 belongs to CDNA4/gfx950 MI350-series hardware. MI300X FP4 support will mean
packed weight-only storage plus dequantization, not a false native-FP4 claim.
