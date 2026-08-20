# 2026-08-20 — Experiment 003 parallel HIP RMSNorm

## Problem

One GPU thread processed a whole RMSNorm row. After the first two experiments, that
simple reference Kernel consumed 64.31% of Qwen training Kernel time.

## Change

- one 256-thread block processes one row;
- FP32 shared-memory reductions compute square sum and backward weighted dot;
- a device scratch Tensor holds one inverse RMS per row;
- a second Kernel reduces weight gradient by column without atomics;
- CPU formula, public API, dtype and model graph remain unchanged.

## Correctness

```text
CPU debug                    150/150 pass
ASan/UBSan                   148/148 pass
HIP release                   42/42 pass
PyTorch operator parity         4/4 pass
rows                            1/3/32
widths                  16/384/512/896/1536
focused payload transfer       H2D=0, D2H=0
```

Official Qwen and DeepSeek generated token sequences remain exact. Parallel reduction
changes summation order, so small FP32 logit/loss differences are expected and remain
inside the existing external-reference tolerance.

## End-to-end result

| Workload | Before | After | Change |
|---|---:|---:|---:|
| Qwen train | 38.772 | 71.057 token/s | 1.83× |
| Qwen generate | 35.355 | 57.322 token/s | 1.62× |
| DeepSeek train | 22.356 | 47.913 token/s | 2.14× |
| DeepSeek generate | 10.145 | 18.597 token/s | 1.83× |

```text
score  0.479227 → 0.885816
gain   84.8%
```

Training ratios above PyTorch apply only to this fixed batch-1, three-predicted-token,
FP32 experiment. They are not a broad long-context training claim.

## Profiler

```text
RMSNorm Kernel time       75.85 ms → 1.55 ms
RMSNorm Kernel share       64.31% → 3.59%
all Kernel time          117.94 ms → 43.25 ms
profiled Qwen step         99.8 ms → 57.6 ms
```

Generation is now the lower side of the four-workload score, so the next experiment is
the device KV/GQA data path rather than another training-only Kernel.

Decision: `keep`.
