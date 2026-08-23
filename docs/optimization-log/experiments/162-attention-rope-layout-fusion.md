# Experiment 162 — fuse the Attention projection layout into bias + RoPE

## Problem

Experiment 161's Runtime diagnostics reduced every remaining strided materialization to
four Attention layouts. Q/K projection output is naturally contiguous `[B,T,H,D]`, but the
old graph created a transpose view, copied it to `[B,H,T,D]`, ran bias + split-half RoPE,
then repeated the reverse copy during backward. Bias-gradient preparation made another
`[B,H,T,D] → [B,T,H,D]` copy.

The mathematics did not require any of those intermediate Tensors. Attention consumes
`[B,H,T,D]`, while the preceding projection and its gradient consume `[B,T,H,D]`.

## Candidate

`rope_split_half_bias_bthd` reads contiguous `[B,T,H,D]`, adds `[H*D]` bias, rotates each
split-half pair and writes contiguous `[B,H,T,D]` in one HIP launch. Its backward reads
the Attention gradient in `[B,H,T,D]`, applies inverse rotation and writes `[B,T,H,D]`.
That result reshapes to `[B*T,H*D]` without copying for `bias_gradient`.

The graph-level execution switch
`--attention-rope-layout-fusion true/false` changes only this layout route. Default is
enabled. The old materialized path remains available as the same-binary rebuttal.

## Correctness gates

- CPU operator forward and backward match the old composed layout path, including distinct
  `T=4, H=2` dimensions, nonzero position offset and nondefault base;
- the dedicated graph test compares forward, input gradient and bias gradient while proving
  that the fused node has no transpose/contiguous parents;
- the independent Python/PyTorch graph reconstructs bias broadcast, transpose and split-half
  RoPE, then compares output and both gradients;
- HIP operator output/backward match CPU with zero host payload transfers;
- a complete CPU/HIP Transformer compares loss and every parameter gradient.

## Same-binary official T512 A/B

The runner supplies 513 raw tokens so next-token shifting leaves exactly 512 trained
positions. Each policy/model has three fresh processes; policy order alternates. Training
uses BF16 Linear forward, FP32 masters, persistent mirrors, B1, one warm-up and two measured
steps.

| Model | Materialized | Fused | Throughput | Peak bytes saved | Allocation calls saved |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,711.83 | 14,705.47 tok/s | 0.9996× | 48,234,496 | 288 |
| DeepSeek Distill 1.5B | 6,095.45 | 6,158.82 tok/s | 1.0104× | 102,760,448 | 336 |

Qwen final-loss relative difference is 0.2351%; DeepSeek is exact at the reported
precision. The fixed observed parameter is equal for both models. Both throughput ratios
clear the declared 0.98 non-regression gate.

Diagnostics show the intended structural change:

| Model | Materialized strided copies | Fused | Byte reduction |
|---|---:|---:|---:|
| Qwen | 240 / 251,658,240 B | 96 / 100,663,296 B | 60.0% |
| DeepSeek | 280 / 513,802,240 B | 112 / 205,520,896 B | 60.0% |

The remaining 40% is Value layout plus the context output transpose; this candidate does
not claim to eliminate those boundaries.

## rocprofv3 attribution

On the same Qwen workload, Kernel dispatches fall `7,624→7,192`, strided-copy calls
`720→288`, and strided-copy time `3.656→1.471 ms`. The new forward/backward RoPE kernels
are individually about 1–2% slower because they calculate two layout indices, yet total
Kernel time falls `112.218→110.511 ms` (`1.52%`). Instrumented throughput is deliberately
not used as the performance result.

![Attention RoPE layout fusion](../assets/attention-rope-layout-fusion.svg)

## Decision

Keep as a graph/layout and memory optimization. It removes 60% of attributed strided-copy
traffic, lowers both official-model peaks, is throughput-neutral on Qwen and positive on
DeepSeek, and passes independent forward/backward gates. The result also narrows the next
layout target: Value input and context-output transpose, not Q/K RoPE.

Raw evidence is in
[`benchmarks/results/2026-08-23-attention-rope-layout-fusion/`](../../../benchmarks/results/2026-08-23-attention-rope-layout-fusion/).
