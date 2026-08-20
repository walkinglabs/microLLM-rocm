# Experiment 001 raw evidence

## Files

| File | Meaning |
|---|---|
| `microllm.jsonl` | optimized Qwen/DeepSeek 2 warm-up + 5 measured raw rows |
| `comparison.jsonl` | ratios against fixed experiment-000 PyTorch raw baseline |
| `before-kernel-stats.csv` | Qwen train baseline rocprof Kernel statistics |
| `after-kernel-stats.csv` | Qwen train optimized rocprof Kernel statistics |
| `before-hip-api-stats.csv` | baseline HIP API statistics |
| `after-hip-api-stats.csv` | optimized HIP API statistics |
| `before-allocation-stats.csv` | baseline memory allocation call statistics |
| `after-allocation-stats.csv` | optimized memory allocation call statistics |

Large PFTrace/JSON trace files remain outside Git. Commands and retained temporary
locations are recorded in the parent experiment report.

## Headline

```text
CE Kernel share                 75.73% → approximately 0.62%
Qwen measured train            7.30 → 24.03 token/s
DeepSeek measured train        5.79 → 13.30 token/s
geometric parity score         0.191660 → 0.318328
```

Generation remained within normal run variation because this experiment did not change
the inference graph.
