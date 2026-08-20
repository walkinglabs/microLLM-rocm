# Experiment 006 raw evidence

## Files

| File | Meaning |
|---|---|
| `audit-before-pool.jsonl` | call-count instrumentation with ordinary hipMalloc/hipFree |
| `rejected-early-enable.jsonl` | faster but over-retaining first pool candidate |
| `microllm.jsonl` | kept steady-state pool, 2 warm-up + 5 measured rows |
| `comparison.jsonl` | final ratios against fixed Experiment 000 PyTorch data |
| `after-*-stats.csv` | kept candidate's compact Qwen generation rocprof tables |

The direct profiler before tables are Experiment 005's
[`after-*`](../005-data/README.md). The large final PFTrace stays outside Git at
`/tmp/microllm-qwen-infer-profile-exp006/`.

## Headline

```text
Qwen generation       93.34 → 134.87 token/s
DeepSeek generation   38.99 → 48.93 token/s
Qwen train            72.33 → 107.08 token/s
DeepSeek train        49.47 → 69.77 token/s
score                 1.219170 → 1.700597
```
