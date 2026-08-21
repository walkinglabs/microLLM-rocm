# Experiment 052 — split non-atomic K/V backward (discarded)

## Hypothesis

The retained backward atomically adds K/V gradients from every query row. At T=512 it
occupies 50.64% of Kernel time. A two-stage algorithm might trade a small workspace for
conflict-free writes:

1. each query row writes its probability and score-gradient row and computes Q gradient;
2. each K/V output element owns its destination and sums future query rows without atomics.

For Qwen the two FP32 `[B,H,T,T]` workspaces total 29,360,128 bytes per layer invocation.

## Correctness

MHA and GQA at the T=256 dispatch boundary matched CPU for Q, K and V gradients, including
causal future masking. There were no payload transfers. T≤128 retained the atomic path.

## Result

| Metric | Experiment 051 | Split candidate | Ratio |
|---|---:|---:|---:|
| Qwen T=512 throughput | 812.45 tok/s | 688.82 tok/s | 0.848× |
| measured peak | 12.673 GB | 12.673 GB | 1.000× |
| dispatches in process trace | 6,767 | 6,695 | 0.989× |
| Attention backward time | 985.61 ms | 1320.85 ms | 0.746× speedup |

![Split K/V backward discarded](../assets/split-kv-backward-discard.svg)

The row stage costs 478.27 ms and the new K/V reducer costs 842.58 ms. Avoiding atomics
does not remove work: every K/V output thread now scans query positions and repeated heads,
while the T×T matrices are written and read again. Backward becomes 34% slower.

## Decision

Discard the code. A successful next design must use tiled matrix operations or a flash-style
backward that reuses score tiles; another scalar output-thread rescan is already falsified.
