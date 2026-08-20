# 2026-08-20 — Experiment 002 transpose-aware GEMM

## Problem

The engine used to implement `A × Bᵀ` in two separate actions:

```text
copy B into a new transposed Tensor
→ multiply A by the copied Tensor
```

For Qwen's tied output weight, that temporary copy is roughly 544 MB. Backward repeated
the same pattern for Linear weights.

## Change

- added a contiguous 2D `op(A) × op(B)` contract;
- added matching CPU, readable HIP and hipBLASLt paths;
- used the contract in 2D autograd backward and the tied output head;
- kept the original batched/non-contiguous matmul path for compatibility;
- did not cache a duplicate weight or add a global synchronization.

## Correctness

```text
CPU debug                    150/150 pass
ASan/UBSan                   148/148 pass
HIP release                   41/41 pass
PyTorch operator parity         4/4 pass
HIP dtype/layout matrix       FP32/FP16/BF16 × NN/NT/TN/TT
device payload transfers      H2D=0, D2H=0 during focused GEMM
```

The graph gate compares tied-head output, hidden gradient and shared embedding-weight
gradient. It also asserts that the forward graph contains no transpose/materialization
node. Official Qwen and DeepSeek generated tokens remain unchanged.

## End-to-end result

| Workload | Before | After | Change |
|---|---:|---:|---:|
| Qwen train | 24.027 | 38.772 token/s | 1.61× |
| Qwen generate | 18.847 | 35.355 token/s | 1.88× |
| DeepSeek train | 13.295 | 22.356 token/s | 1.68× |
| DeepSeek generate | 10.053 | 10.145 token/s | 1.01× |

```text
score  0.318328 → 0.479227
gain   50.5%
```

## Profiler

```text
strided-copy calls          1302 → 624
strided-copy time       62.33 ms → 2.16 ms
all Kernel time        185.76 ms → 117.94 ms
profiled Qwen step       149.1 ms → 99.8 ms
```

RMSNorm is now the dominant Qwen training Kernel group at 64.31%. That result chooses
Experiment 003; it does not prove every remaining strided copy should be removed.

## Evidence

The experiment contract, raw Qwen/DeepSeek JSONL, comparison rows and compact rocprof
tables live in `docs/optimization-log/experiments/002-*`.

Decision: `keep`.
