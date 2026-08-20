# Step 00 — fixed FP32 baseline

Status: `complete`

## Purpose

建立所有后续实验不可偷偷改变的起点。

## Fixed workload

```text
MI300X gfx942
FP32
Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B
2 warm-up + 5 measured
same official checkpoints and token IDs
```

## Measured result

| Workload | Ratio |
|---|---:|
| Qwen train | 0.142235 |
| Qwen generate | 0.267461 |
| DeepSeek train | 0.220921 |
| DeepSeek generate | 0.160553 |

Geometric score: `0.191660`.

## Profiler facts

- train CE forward/backward: 75.7% Kernel time;
- inference tied transpose + RMSNorm: 81.1%;
- Qwen inference: 7407 allocations, 7403 frees;
- Qwen training: 5420 allocations, 5416 frees;
- AdamW: approximately 1.5% of microLLM Qwen training Kernel time.

## Evidence

- `benchmarks/results/2026-08-20-mi300x-pytorch-hf-comparison/`
- `docs/development/2026-08-20-pytorch-performance-comparison.md`
- local rocprof baseline retained outside Git due PFTrace size.

## Decision

`baseline`. No performance conclusion is generalized beyond the fixed matrix.
