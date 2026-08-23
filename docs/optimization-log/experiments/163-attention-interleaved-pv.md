# Experiment 163 — prove interleaved-head P×V before changing the graph

## Why the first context idea was rejected

Fusing BHTD→BTHD only into the output-projection BF16 cast looks attractive in forward.
Backward weight-gradient still needs the original FP32 context in logical BTHD order. It
would recreate the same full materialization, so the proposal only moved the copy from
forward to backward.

The correct seam is earlier: make Attention produce BTHD directly and later make its
backward accept BTHD. This experiment tests the required forward GEMM independently.

## Layout contract

The operator computes:

```text
probabilities [B,H,T,T]
× value       [B,T,H,D]
→ context     [B,T,H,D]
```

For a fixed head, value and context remain `T×D` matrices. hipBLASLt uses `ld=H*D` and
`batch_stride=D`, so heads occupy different D-wide slices inside every token row. For
multiple outer batches, one H-head batched call is submitted per batch.

CPU uses the explicit transpose/matmul/transpose reference. HIP uses the custom matrix
layouts. PyTorch independently evaluates the same expression. Shape tests include B2,
distinct T/H, invalid head count and zero host transfers.

## MI300 matrix

Five shapes, two implementations and three fresh processes produce 30 complete-output
rows. Each process uses three warm-ups and twenty measured repetitions. Policy order
alternates to reduce drift.

| Shape B×H×T×D | Event speedup | Wall speedup | Max/RMS error |
|---|---:|---:|---:|
| 2×2×3×2 | 1.006× | 1.003× | 0 / 0 |
| 2×14×32×64 | 1.056× | 1.062× | 0 / 0 |
| 1×14×128×64 | 1.316× | 1.260× | 0 / 0 |
| Qwen 1×14×512×64 | 1.415× | 1.330× | 0 / 0 |
| DeepSeek 1×12×512×128 | 2.200× | 2.136× | 0 / 0 |

The edge case is intentionally retained: eliminating copies is nearly neutral when the
matrix is tiny. The benefit grows with context and width, exactly as a memory-layout
hypothesis predicts.

![Interleaved Attention P×V](../assets/attention-interleaved-pv.svg)

## Decision

Keep the primitive. It proves the backend supports the nonstandard layout, passes exact
complete-output gates and exceeds the 1.05 official T512 operator gate. It does not enter
the model until corresponding BTHD backward GEMMs and full graph gradients exist.

Raw evidence is in
[`benchmarks/results/2026-08-23-attention-interleaved-pv/`](../../../benchmarks/results/2026-08-23-attention-interleaved-pv/).
