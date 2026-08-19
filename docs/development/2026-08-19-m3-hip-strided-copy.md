# 2026-08-19 — M3 HIP strided-to-contiguous copy

## Contract

Materialize a non-contiguous float32 HIP Tensor in logical row-major order without a
per-element host copy. Preserve CPU behavior and reject unsupported layouts visibly.

## Implementation

- one HIP thread maps one logical output index to a source storage index;
- shape and stride metadata up to rank eight are passed by value in a fixed layout;
- Tensor `contiguous()` dispatches the same semantic operation to CPU or HIP;
- non-contiguous HIP `to_vector()` first materializes on device, then performs one
  device-to-host copy;
- shape overflow, rank mismatch, null pointers, and rank above eight fail.

## Verification

A `2x3` HIP Tensor is transposed as a zero-copy `3x2` view, materialized by the new
kernel, and returns logical values `0,3,1,4,2,5`. CPU Tensor tests remain green.

This kernel removes a blocker for transposed matrices in HIP backward and for
multi-token cached layouts. It is readable generic indexing, not a tuned transpose.
