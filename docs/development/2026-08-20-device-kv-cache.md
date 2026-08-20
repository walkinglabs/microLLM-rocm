# 2026-08-20 — Experiment 004 device KV Cache and direct GQA

## Problem

Every cached token copied K/V to the CPU, built a larger Tensor, copied it back, and
physically repeated KV heads for GQA. The Storage address changed as the prefix grew.

## Kept design

- preallocate K/V to the current request bound;
- write one position in place on the device;
- expose only the initialized prefix through Tensor shape/stride metadata;
- map query heads to KV heads inside cached score/context Kernels;
- keep training Attention and CPU sampling unchanged.

An initial maximum-model-context allocation was rejected after DeepSeek peak memory rose
from about 7.11 GB to 14.63 GB.

## Evidence

```text
CPU debug                    151/151 pass
ASan/UBSan                   149/149 pass
HIP release                   42/42 pass
PyTorch operator parity         4/4 pass
focused cached decode          H2D=0, D2H=0
Qwen generate             57.32 → 85.64 token/s
DeepSeek generate         18.60 → 35.79 token/s
score                     0.885816 → 1.167931
```

The Qwen 1/32/128/512-token curve and matched before/after rocprof tables are committed
under `docs/optimization-log/experiments/004-data/`.

Decision: `keep`.
