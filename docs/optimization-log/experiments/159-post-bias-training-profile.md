# Experiment 159 — post-bias training phase profile

## Why another profile

The ordinary rocprofv3 process trace includes checkpoint loading, warm-up and measured
training. After Experiment 158, its fourth-largest named Kernel was BF16-to-FP32
cast-transpose. Optimizing it would not necessarily change training throughput.

## Phase subtraction

The same candidate binary runs two Qwen T512 profiles:

```text
load + 0 warm-up + 1 measured step
load + 1 warm-up + 2 measured steps
```

For every exact Kernel name, `(three-step calls/time - one-step calls/time) / 2`
estimates one additional training step. Only positive call deltas enter the result.
BF16-to-FP32 cast-transpose has no positive delta, proving its 168 calls are load-only.

## Derived bottleneck map

| Category | ms/step | Kernel share |
|---|---:|---:|
| hipBLASLt GEMM | 18.98 | 53.47% |
| AdamW | 5.66 | 15.95% |
| other kernels | 3.11 | 8.76% |
| gradient/elementwise add | 1.41 | 3.96% |
| cross entropy | 1.32 | 3.73% |
| cooperative bias gradient | 1.32 | 3.72% |
| strided materialization | 1.29 | 3.63% |
| RMSNorm forward/backward | 1.18 | 3.32% |
| FP32/BF16 cast | 0.95 | 2.67% |

![Post-bias training profile](../assets/post-bias-training-profile.svg)

## Decision

Do not optimize cast-transpose for a training-throughput claim. Do not reopen current
AdamW Vectorized/chunked routes without a new design; Experiment 157 already falsified
their model gate. The largest open category is training GEMM at 53.47%.

The next implementation node will extend correctness-before-timing matmul work from
implementation selection to hipBLASLt solution-index enumeration for exact training
shapes, then require complete output/gradient and two-model end-to-end gates.

Raw inputs and the machine-derived delta are in
[`benchmarks/results/2026-08-23-post-bias-training-profile/`](../../../benchmarks/results/2026-08-23-post-bias-training-profile/).
