# Experiment 018 — wide residual-Norm block size

Status: `keep`

## Failure carried forward

Experiment 017 improved Qwen but regressed DeepSeek generation 4.2%. The fused Kernel
used 256 threads for both hidden widths: Qwen 896 and DeepSeek 1536.

## Hypothesis

At width 1536, 512 threads reduce each thread's input work and expose more parallelism.
Width 896 keeps 256 threads, so the experiment changes only the wide row.

## Scope and falsification

- same FP32 equation, outputs, cache and public API;
- `width >= 1024` launches 512 threads; narrower rows launch 256;
- no Qwen or training path change;
- reject if the target Kernel or three-process DeepSeek median does not improve;
- do not generalize the threshold beyond measured widths.

## Correctness

- focused fused-op and cached GQA tests pass;
- full MI300X/gfx942 HIP suite: `56/56` pass;
- official DeepSeek top logits and all eight generated token IDs remain exact;
- CPU/sanitizer/PyTorch contracts are unchanged from Experiment 017.

## Three-process result

```text
DeepSeek samples        53.59 / 58.50 / 58.32 token/s
candidate median        58.32 token/s
Experiment 017 median   53.20 token/s
change                  +9.6%
PyTorch ratio           0.934716×
score                   1.803226 → 1.845199
```

The target Kernel's profiled average falls from 6.46 to 4.83 microseconds, about 25%.
Instrumented total decode remains noisy (`29.74 → 29.27 token/s`) because unrelated GEMM
duration changed; the uninstrumented median and target-Kernel timing are the decision
evidence.

## Decision

`keep`. This is an explicit two-region width policy backed by two official hidden widths,
not a claim that 512 threads is universally best. More widths belong in a future operator
matrix before changing the 1024 boundary.

Raw evidence is in [018-data](018-data/README.md).
