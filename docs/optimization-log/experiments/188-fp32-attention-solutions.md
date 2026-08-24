# Experiment 188 — FP32 Attention has faster vendor solutions

Status: `keep` operator candidates; model registration pending

## Question

Persistent Storage removed the measured largest allocation source without moving end-to-end speed.
Can exact hipBLASLt algorithms reduce the actual FP32 QK/PV device time?

## Correctness-before-timing contract

The standalone tuner recreates the framework's row-major and strided-batch descriptors. QK uses
`[H,T,D]×[H,T,D]ᵀ→[H,T,T]`; PV uses `[H,T,T]×[H,T,D]→[H,T,D]`. The model/head dimensions are fixed
from the official configs.

Every heuristic candidate runs once, copies the complete output and checks finite/Max/RMS against
the default same-revision solution. Only passing candidates receive two warm-ups and five Event/
wall measurements. Three fresh processes are intersected by solution index; recommendation uses
the median Event P50, not one lucky sample.

## Formal result

| Model | Op | Passing common indices | Default | Recommended | Index | Speedup |
|---|---|---:|---:|---:|---:|---:|
| Qwen | QK | 64 | 0.021128 ms | 0.015956 ms | 305434 | 1.324× |
| Qwen | PV | 64 | 0.017720 ms | 0.014794 ms | 294519 | 1.198× |
| DeepSeek | QK | 64 | 0.023253 ms | 0.018562 ms | 305460 | 1.253× |
| DeepSeek | PV | 64 | 0.021088 ms | 0.018923 ms | 292941 | 1.114× |

All four pass the 1.05 operator gate. Recommended Max/RMS is bounded by
`4.47035e-7` / `6.6408e-8`; workspace is zero.

![FP32 Attention solutions](../assets/fp32-attention-solutions.svg)

## Decision

Keep the tuner, runner and four exact candidates. Do not write indices into default dispatch in
this node: version-local indices require an explicit exact-shape registry and full logits/model
performance gate. Experiment 189 may register only these four shapes and must compare against the
retained FFN-selective baseline.

Raw evidence:
[`benchmarks/results/2026-08-24-fp32-attention-solutions/`](../../../benchmarks/results/2026-08-24-fp32-attention-solutions/).
