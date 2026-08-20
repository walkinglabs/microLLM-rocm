# Experiment 007 — hipBLASLt descriptor/layout plan cache

Status: `discard`

## Observed bottleneck

After Experiment 006, projection GEMMs dominate the Qwen generation Kernel trace.
Every regular hipBLASLt call still constructs and destroys one operation descriptor and
three matrix layouts even when shape/dtype/transpose flags repeat hundreds of times.

## Hypothesis

An immutable, exact-key, thread-local plan cache will remove repeated host setup without
changing GEMM selection or numerical results. It should improve generation if descriptor
setup is material relative to short GEMMs.

The hypothesis is falsified if cache hits are high but fixed end-to-end throughput does
not improve beyond normal variation.

## Scope

- allowed: regular FP32/FP16/BF16 hipBLASLt descriptors/layouts and cache statistics;
- unchanged: FP8 dynamic-scale descriptor, GEMM algorithm, workspace, registry choice,
  kernels, model, allocator and workload;
- exact key includes dtype, physical shapes and both transpose flags;
- thread-local ownership avoids mutable cross-thread descriptor sharing.

## Required gates

- [x] miss once then hit for an identical key
- [x] distinct NN/NT/TN/TT and dtype keys
- [x] clear/reset behavior
- [x] existing readable/hipBLASLt numerical matrix
- [x] exact Qwen/DeepSeek tokens/loss/update
- [x] fixed performance matrix

## Candidate

- immutable plan held one operation descriptor and three matrix layouts;
- cache key contained dtype, both physical shapes and NN/NT/TN/TT flags;
- cache and hit/miss counters were thread-local;
- FP8 stayed uncached because its descriptor owns dynamic scale pointers;
- focused tests observed one miss followed by one hit, five distinct keys, numerical
  parity and clean reset.

## Fixed performance result

| Workload | Running best | Candidate | Change | Candidate PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 107.080 token/s | 112.314 token/s | +4.9% | 2.188328 |
| Qwen generate | 134.868 token/s | 126.644 token/s | -6.1% | 1.804502 |
| DeepSeek train | 69.770 token/s | 66.145 token/s | -5.2% | 2.522112 |
| DeepSeek generate | 48.929 token/s | 48.701 token/s | -0.5% | 0.780506 |

```text
running-best score  1.700597
candidate score     1.669755
relative change       -1.8%
```

Qwen generation and DeepSeek training both cross the unexplained 5% regression line.
The candidate also changes exact allocation/Event readiness timing, so isolated host
object savings cannot be inferred from a noisy positive training row.

## Correctness

The focused FP32/FP16 and NN/NT/TN/TT gates pass. Official generated tokens, final loss
and updates remain unchanged. Full regression was not rerun after the end-to-end keep
gate failed; the candidate source was removed and `main` returned byte-for-byte to the
Experiment 006 implementation before this report was committed.

## Evidence

Raw candidate JSONL and PyTorch comparison rows are in [007-data](007-data/README.md).
No profiler claim is made because the predefined end-to-end gate already rejected the
candidate.

## Results

Falsified: descriptor/layout reuse was functionally correct but did not improve the
fixed end-to-end objective.

## Decision

`discard`. No plan-cache API or implementation remains in framework source.
