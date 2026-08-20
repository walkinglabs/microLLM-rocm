# Experiment 003 raw evidence

## Files

| File | Meaning |
|---|---|
| `microllm.jsonl` | Qwen/DeepSeek, 2 warm-up + 5 measured raw rows |
| `comparison.jsonl` | ratios against fixed Experiment 000 PyTorch raw data |
| `after-kernel-stats.csv` | Qwen train rocprof Kernel statistics after Experiment 003 |
| `after-hip-api-stats.csv` | HIP API statistics after Experiment 003 |
| `after-allocation-stats.csv` | allocation/free statistics after Experiment 003 |

Experiment 002's [`after-*`](../002-data/README.md) files are the direct before tables.
The large PFTrace is not committed; its replay command and temporary location are in
the parent report.

## Headline

```text
RMSNorm time                75.85 ms → 1.55 ms
Qwen measured train         38.77 → 71.06 token/s
Qwen measured generation    35.35 → 57.32 token/s
DeepSeek measured train     22.36 → 47.91 token/s
DeepSeek generation         10.15 → 18.60 token/s
geometric parity score      0.479227 → 0.885816
```
