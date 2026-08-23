# Experiment 165 — choose a new hotspot after layout copies reach zero

## Method

A full profiler run includes weight loading and the first lazy library setup. We collect:

```text
profile A = load + 1 training step
profile B = load + 3 training steps
per-step = (B - A) / 2, matched by exact Kernel name
```

This is the same phase-delta method as Experiment 159, now on the retained BTHD graph.

## Current per-step Kernel map

| Category | Time | Share | Calls |
|---|---:|---:|---:|
| hipBLASLt GEMM | 18.858 ms | 56.55% | 651 |
| AdamW | 5.590 ms | 16.76% | 290 |
| other kernels | 3.165 ms | 9.49% | 411 |
| bias gradient | 1.376 ms | 4.13% | 72 |
| cross entropy | 1.307 ms | 3.92% | 5 |
| RMSNorm forward/backward | 1.191 ms | 3.57% | 147 |
| FP32/BF16 cast | 0.976 ms | 2.93% | 168 |
| gradient/elementwise add | 0.859 ms | 2.58% | 168 |
| fill | 0.027 ms | 0.08% | 1 |

Total attributable Kernel time is `33.349 ms/step`, down from Experiment 159's
`35.497 ms/step`. Strided materialization no longer has a positive category. Load-only
BF16→FP32 cast and cast-transpose again remain in the excluded list.

## Why the largest bars are not automatically next

GEMM is largest, but Experiment 160 screened 1,536 complete-output candidates and rejected
both all-shape and selective model policies. AdamW is the largest single Kernel name, but
Experiment 157 found no aligned candidate over its gate and kept Scalar. Repeating those
same searches without a new boundary would not be research.

The new BTHD Attention path has a different host boundary. Each layer calls interleaved
P×V, dP and dV once. Each call currently constructs three hipBLASLt matrix layouts plus one
matmul description. That is 72 calls/step for 24-layer Qwen and 84 for 28-layer DeepSeek.

Experiment 163 measured interleaved operator wall/Event medians:

```text
Qwen     0.040581 - 0.031231 = 0.009350 ms/call → about 0.673 ms/step
DeepSeek 0.174595 - 0.165818 = 0.008777 ms/call → about 0.737 ms/step
```

This extrapolation includes general host timing and is not proof that descriptors alone
cost that amount. It is only strong enough to justify an exact immutable plan-cache
experiment with a same-revision model rebuttal.

![Post-layout training profile](../assets/post-layout-training-profile.svg)

## Decision

Close further work on the diagnosed strided-copy set. Do not retry generic AdamW or
standard BF16 solution policies. The next candidate may only cache interleaved Attention
descriptor/layout state, must preserve all outputs, and must clear a 1.01 throughput gate
on both official models without increasing peak memory.

Raw evidence is in
[`benchmarks/results/2026-08-23-post-layout-training-profile/`](../../../benchmarks/results/2026-08-23-post-layout-training-profile/).
