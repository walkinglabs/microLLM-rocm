# Experiment 024 — retirement batch 16 → 32

Status: `discard`

## Question

Would halving Event frequency again improve on the measured 16-block allocator batch?
Only the batch constant changed; focused lifetime and stress tests passed.

## Three-process medians

| Workload | Batch 16 | Batch 32 | Change |
|---|---:|---:|---:|
| Qwen train | 158.85 | 159.50 | +0.4% |
| Qwen generate | 218.40 | 207.90 | -4.8% |
| DeepSeek train | 81.09 | 83.01 | +2.4% |
| DeepSeek generate | 78.10 | 78.71 | +0.8% |

```text
score       2.470863 → 2.462231
```

Backend allocations also rise because blocks wait longer for a full retirement batch.
The candidate does not beat the running best and introduces a meaningful Qwen generation
regression.

## Decision

`discard`. Restore 16 as the measured local optimum. “Fewer Events” stops being useful
when delayed reuse outweighs the saved API calls.

Raw evidence is in [024-data](024-data/README.md).
