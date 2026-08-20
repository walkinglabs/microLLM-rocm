# 2026-08-20 — Experiment 005 device greedy sampling

## Change

A block-parallel FP32 argmax returns one int32 Tensor on the GPU. The generator copies
that scalar to its C++ result vector but reuses the device Tensor for the next Embedding.
Stochastic top-k/temperature stays on the existing CPU reference path.

## Evidence

```text
CPU debug                    152/152 pass
ASan/UBSan                   150/150 pass
HIP release                   44/44 pass
PyTorch operator parity         4/4 pass
vocabularies              32/8192/151936
selected payload               4 bytes/token
Qwen generate             85.64 → 93.34 token/s
DeepSeek generate         35.79 → 38.99 token/s
score                     1.167931 → 1.219170
```

The matched profiler shows generated-loop D2H records falling from 9 to 1. The one
remaining record belongs to the app's separate pre-generation full-logit report.

Decision: `keep`.
