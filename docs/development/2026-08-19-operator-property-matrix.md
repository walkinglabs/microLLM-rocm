# 2026-08-19 — operator shape and property matrix

## Why this was added

A single `2 x 3` example can pass even when an operator is wrong for a scalar, a
four-dimensional Tensor, a zero inner dimension, or a large softmax input. Fixed
PyTorch examples remain the external oracle, but they are not enough to exercise many
shapes. This milestone adds a second kind of evidence: properties that must remain true
over a deterministic matrix of shapes and values.

## Implemented checks

`tests/ops/property_test.cpp` uses fixed random seeds so every failure can be repeated.
It checks:

- add, multiply, and scale from rank zero through rank four against scalar loops;
- 2D, batched 3D/4D, narrow, and zero-inner-dimension matmul against an independent
  nested-loop reference;
- softmax finiteness, non-negativity, and row sums, including inputs `80` and `-80`;
- RMSNorm, SiLU, and SwiGLU values over widths 1, 3, 5, and 8;
- scalar/1D/2D embedding index shapes and exact gathered rows;
- causal masking for sequence lengths 1, 2, and 5;
- cross-entropy over rank 2 and rank 3 logits;
- RoPE pair-norm preservation for three ranks and head widths 2, 4, and 8;
- full gradients of an RMSNorm → SwiGLU → sum graph at widths 1, 2, and 5 against
  central finite differences.

These checks complement rather than replace the same-value PyTorch oracle and the
CPU/HIP comparison. A property can reveal a family of bugs, while the external oracle
guards against both implementations making the same mistaken assumption.

## Measured result

```text
normal CPU CTest    121/121 pass
CPU ASan/UBSan      119/119 pass
coverage audit      33 registered test files pass
```

The sanitizer preset intentionally excludes the C and Python dynamic binding tests;
those run separately because preloading sanitizer runtimes into foreign interpreters
would test loader ordering rather than the framework code.
