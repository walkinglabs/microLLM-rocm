# Experiment 004 raw evidence

## Files

| File | Meaning |
|---|---|
| `microllm.jsonl` | final Qwen/DeepSeek 2 warm-up + 5 measured rows |
| `comparison.jsonl` | ratios against the fixed Experiment 000 PyTorch data |
| `qwen-context-curve.jsonl` | 1/32/128/512 new-token measurements |
| `before-*-stats.csv` | detached Experiment 003 Qwen generation rocprof tables |
| `after-*-stats.csv` | kept Experiment 004 Qwen generation rocprof tables |

The before binary was built from detached commit `650ab51`; its temporary source
worktree was removed after the compact tables were copied. Large profiler traces remain
outside Git at the paths recorded in the parent experiment report.

## Headline

```text
Qwen measured generation      57.32 → 85.64 token/s
DeepSeek measured generation  18.60 → 35.79 token/s
hipMemcpy calls                 2712 → 600
geometric parity score       0.885816 → 1.167931
```
