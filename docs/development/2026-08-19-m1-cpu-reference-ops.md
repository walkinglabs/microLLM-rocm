# 2026-08-19 — M1 CPU reference operators

## Contract

Implement readable CPU float32 oracles before HIP kernels. Preserve logical Tensor
order, expose shape errors, and include hand-computable expected values. Do not add
performance claims or silently transfer HIP tensors to the CPU.

## Implemented operators

- add, elementwise multiply, scale;
- equal-batch-rank matrix multiplication;
- int32 embedding lookup;
- stable last-dimension softmax;
- RMSNorm;
- SiLU and SwiGLU;
- RoPE with an explicit sequence dimension and position offset;
- mean cross entropy with int32 targets.

Tensor gained explicit int32 vector construction/materialization for token indices
and labels. This is dtype-specific rather than an unchecked template interface.

## Verification

```text
CPU Debug:        30/30 passed
CPU ASan/UBSan:   30/30 passed
HIP-enabled build: 30 passed, 1 N0-only test skipped
```

The cases include manually calculated batched matmul, stable softmax near 1000,
RMSNorm, RoPE at positions zero and one, stable cross entropy, invalid shapes, and
out-of-range embedding indices.

## Current boundary

These are deliberately reference implementations. They allocate readable outputs
and do not claim production CPU speed. HIP input fails rather than performing a
hidden device-to-host fallback. The next change implements readable HIP kernels and
compares each result with this oracle.
