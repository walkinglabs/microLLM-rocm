# 2026-08-19 — M1 readable HIP Transformer operators

## Contract

Complete the first readable HIP forward set using the same public semantics as the
CPU oracle. Each GPU result must return through an explicit Tensor transfer and match
the CPU value within a stated tolerance. No tuned-kernel performance claim is made.

## Added HIP paths

- int32 Embedding;
- stable last-dimension Softmax;
- RMSNorm;
- SiLU and SwiGLU;
- RoPE with explicit sequence dimension and position offset;
- mean cross entropy.

Together with the prior change, the readable HIP set contains fill, add, multiply,
scale, batched matmul, Embedding, Softmax, RMSNorm, SiLU, SwiGLU, RoPE, and cross
entropy.

## Verification

On the visible `gfx942` device:

```text
HIP conformance: 6/6 passed
CPU-labelled regression in HIP build: 30 passed, 1 N0-only test skipped
```

The tests compare every new path with the CPU oracle, including large softmax inputs,
weighted RMSNorm, RoPE position behavior, and int32 targets.

## Deliberately retained limitations

- readable Softmax and RMSNorm assign one thread per row;
- readable cross entropy uses one thread for the complete reduction;
- invalid GPU indices/targets produce a non-finite result rather than a host exception;
- kernels use the default HIP stream until the next operator-context change;
- only contiguous float32 data and int32 indices are accepted.

These limitations make the first path inspectable. They are inputs to profiling and
optimization, not hidden production claims.
