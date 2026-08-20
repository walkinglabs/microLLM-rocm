# Experiment 026 — fuse V bias into paired cache store

Status: `discard`

The candidate skipped the V projection `add_bias` Tensor and added its head-aware bias
while the paired K/V store wrote cache memory. Focused HIP/cache/token tests passed and
official allocation calls fell by 600 for Qwen and 2,660 for DeepSeek.

Three-process medians nevertheless regressed:

```text
Qwen       219.30 → 209.63 token/s  -4.4%
DeepSeek    78.74 →  77.94 token/s  -1.0%
```

Removing an allocation and Kernel is not sufficient evidence of speed. The public API,
Kernel and model policy were removed. Raw data is in [026-data](026-data/README.md).
