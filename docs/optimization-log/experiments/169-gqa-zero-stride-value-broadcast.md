# Experiment 169 — broadcast Value with zero batch stride

## Boundary

The old GQA P×V path creates `expanded_value[B,T,H,D]` before GEMM. For one KV head and
its `repeats` query heads, every batch item should read the same `T×D` V matrix. hipBLASLt
can express that with:

```text
Value matrix batch stride = 0
probability batch stride  = T*T
context batch stride      = D
```

The implementation loops over outer batch and KV head, submits a `repeats`-wide batched
GEMM, and writes the query-head slice directly into context BTHD. CPU/PyTorch use explicit
Value repeat as the independent reference.

## Correctness

- B2/H4/KV2 tests exercise both outer-batch and KV-group pointer offsets;
- all 30 benchmark processes compare every output element;
- worst Max/RMS difference is `7.6e-8/1.4e-8`;
- timed H2D/D2H calls are zero;
- invalid `H != KV*repeats` is rejected;
- installed package consumer calls the public symbol.

## MI300 matrix

| Shape | Repeats | Event speedup | Wall speedup | Interpretation |
|---|---:|---:|---:|---|
| B2 H4 KV2 T3 D2 | 2 | 0.870× | 0.917× | launch/setup dominates |
| Qwen T128 H14 KV2 D64 | 7 | 0.898× | 0.946× | slower |
| Qwen T512 H14 KV2 D64 | 7 | 0.933× | 0.937× | slower |
| DeepSeek T512 H12 KV2 D128 | 6 | 1.633× | 1.603× | strong win |
| MHA T128 H4 KV4 D64 | 1 | 0.654× | 0.726× | required counterexample |

The zero-stride API is numerically valid. It is not a universal optimization: splitting one
H-batched GEMM into `KV` calls costs more than the removed repeat for Qwen width 64 and is
especially wrong for repeats 1. DeepSeek width 128 saves enough Value traffic to win.

![GQA zero-stride Value broadcast](../assets/gqa-zero-stride-value-broadcast.svg)

## Decision

Reject universal/model-default routing. Keep the P×V primitive and open one narrower
experiment: implement matching dP/dQ broadcast layouts, route only `D>=128` GQA, then require
Qwen to remain on its current path while DeepSeek passes complete gradients and end-to-end
speed. If that selective graph fails, close zero-stride GQA entirely.

Raw evidence is in
[`benchmarks/results/2026-08-23-attention-gqa-zero-stride-broadcast/`](../../../benchmarks/results/2026-08-23-attention-gqa-zero-stride-broadcast/).
