# 2026-08-20 — wide residual-Norm launch tuning

The fused cached residual+RMSNorm Kernel now launches 512 threads when the hidden width
is at least 1024 and retains 256 threads below that boundary. This changes DeepSeek's
1536-wide rows while leaving Qwen's 896-wide rows unchanged.

Evidence on MI300X/gfx942:

```text
DeepSeek generation median    53.20 → 58.32 token/s
target Kernel average          6.46 →  4.83 microseconds
full HIP regression                    56/56 pass
official eight token IDs                exact
```

The fixed four-workload score rises to `1.845199`. The 1024 boundary is not described as
a universal optimum: only widths 896 and 1536 have official-model evidence. See
`docs/optimization-log/experiments/018-wide-residual-norm.md` for raw-run and profiler
scope.
