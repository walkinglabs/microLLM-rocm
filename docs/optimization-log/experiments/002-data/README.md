# Experiment 002 raw evidence

## Files

| File | Meaning |
|---|---|
| `microllm.jsonl` | Qwen/DeepSeek, 2 warm-up + 5 measured raw rows |
| `comparison.jsonl` | ratios against the fixed Experiment 000 PyTorch baseline |
| `after-kernel-stats.csv` | Qwen train rocprof Kernel statistics after Experiment 002 |
| `after-hip-api-stats.csv` | HIP API statistics after Experiment 002 |
| `after-allocation-stats.csv` | allocation/free statistics after Experiment 002 |

The direct before tables are Experiment 001's
[`after-*`](../001-data/README.md) tables. Reusing those files avoids storing an
identical second copy while keeping the comparison exact.

Large PFTrace/JSON trace files stay outside Git. The parent experiment report records
their location and the replay command.

## Headline

```text
strided-copy time           62.33 ms → 2.16 ms
Qwen measured train         24.03 → 38.77 token/s
Qwen measured generation    18.85 → 35.35 token/s
DeepSeek measured train     13.30 → 22.36 token/s
geometric parity score      0.318328 → 0.479227
```
