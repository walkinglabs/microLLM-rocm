# Experiment 023 — retirement batch 8 → 16

Status: `keep`

## Question

Experiment 022 proved shared allocator Events. Would sixteen blocks per completion Event
reduce API overhead further without delaying reuse enough to hurt allocations or models?

Only `kRetirementBatchSize` changes. The focused test now retires and reuses all sixteen
blocks across one completion boundary.

## Correctness

- CPU debug `157/157`, sanitizer `155/155`, HIP `57/57` remain green;
- focused 16-block reuse test passes;
- official logits/tokens/loss/parameter updates remain exact;
- engine peak bytes are unchanged.

## Profiler

```text
Event create/record calls       1,124 → 562
Event record API time            1.95 → 1.40 ms
Event destroy calls             1,071 → 535
instrumented DeepSeek           51.22 → 39.28 token/s
```

The profiler run regresses despite fewer Event calls. This contradiction is retained;
the keep decision therefore requires the uninstrumented repeated matrix.

## Three-process medians

| Workload | Batch 8 | Batch 16 | Change | PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 154.78 | 158.85 | +2.6% | 3.0950× |
| Qwen generate | 201.39 | 218.40 | +8.4% | 3.1119× |
| DeepSeek train | 81.99 | 81.09 | -1.1% | 3.0919× |
| DeepSeek generate | 75.24 | 78.10 | +3.8% | 1.2516× |

```text
score       2.389841 → 2.470863
```

Backend allocations rise slightly in some processes because a partial batch waits
longer, but engine peak bytes do not change and no workload crosses the 5% regression
gate.

## Decision

`keep`. The fixed score improves 3.4%, three workloads improve, the one regression is
1.1%, correctness passes and memory peak is unchanged. Sixteen is a measured setting,
not a universal constant; larger batches require a new experiment.

Raw evidence is in [023-data](023-data/README.md).
