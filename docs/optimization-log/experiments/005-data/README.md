# Experiment 005 raw evidence

## Files

| File | Meaning |
|---|---|
| `microllm.jsonl` | final four-workload rows; training rows reused unchanged Experiment 004 measurements |
| `comparison.jsonl` | ratios against fixed Experiment 000 PyTorch data |
| `after-kernel-stats.csv` | device-argmax Qwen generation Kernel table |
| `after-hip-api-stats.csv` | device-argmax HIP API table |
| `after-memory-copy-stats.csv` | device-argmax copy direction table |
| `after-allocation-stats.csv` | device-argmax allocation/free table |

The direct before tables are Experiment 004's
[`after-*`](../004-data/README.md) files. Large trace files stay outside Git at the path
recorded in the parent experiment.

## Headline

```text
Qwen generation         85.64 → 93.34 token/s
DeepSeek generation     35.79 → 38.99 token/s
generated-loop D2H records   9 → 1
score                   1.167931 → 1.219170
```
