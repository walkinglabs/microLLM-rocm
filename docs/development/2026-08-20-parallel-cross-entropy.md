# 2026-08-20 — Experiment 001 parallel HIP CrossEntropy

## Hypothesis

The one-thread CrossEntropy forward/backward kernels caused the dominant Qwen training
hotspot. Parallel row reductions and elementwise gradient generation should improve
official training without changing inference.

## Implementation

- one 256-thread block per row;
- block maximum and exponential-sum reductions;
- parallel final mean reduction;
- backward row statistics, valid-count factor and elementwise gradient Kernel;
- unchanged FP32, ignore-index and mean-reduction semantics;
- device-local scratch on the existing operator Stream.

## Correctness

```text
CPU                         148/148 pass
ASan/UBSan                  146/146 pass
HIP                          40/40 pass
PyTorch operator parity       4/4 pass
rows                         1/3/32
classes                      2/32/8192/151936
graph Tensor payload copies  H2D=0, D2H=0
```

The Python oracle includes a 3×257 case so the block-boundary implementation is checked
against PyTorch, not only against the repository CPU formula.

## End-to-end result

| Workload | Before | After | Change |
|---|---:|---:|---:|
| Qwen train | 7.300 | 24.027 token/s | 3.29× |
| Qwen generate | 18.771 | 18.847 token/s | neutral |
| DeepSeek train | 5.794 | 13.295 token/s | 2.29× |
| DeepSeek generate | 10.018 | 10.053 token/s | neutral |

```text
score  0.191660 → 0.318328
gain   66.1%
```

## Profiler

```text
CrossEntropy Kernel share  75.73% → approximately 0.62%
profiled Qwen step         420.7 ms → 149.1 ms
```

The new top Kernel groups are strided transpose copies (33.55%) and RMSNorm
forward/backward (41.68% combined). This supports moving to transpose-aware GEMM and
parallel RMSNorm rather than further CE tuning.

## Evidence

Committed under:

```text
docs/optimization-log/experiments/001-parallel-cross-entropy.md
docs/optimization-log/experiments/001-data/
```

Decision: `keep`.
