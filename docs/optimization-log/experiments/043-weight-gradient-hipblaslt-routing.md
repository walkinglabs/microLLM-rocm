# Experiment 043 — route wide weight-gradient GEMM to hipBLASLt

Status: `keep`

## Observation and competing explanations

Experiment 042 showed the strange result that context 32 was slower than context 128.
Two explanations were considered:

1. Attention or another sequence-length operation made context 32 unusually slow.
2. The backward GEMM router chose different implementations at reduction sizes 32 and 128.

Paired profiler traces support the second explanation. Before the change, context 32 ran
507 readable `transpose(left)` FP32 matmuls taking 1.228 seconds, 75.75% of all Kernel time.
At context 128 that readable Kernel disappeared because the old `inner >= 128` rule selected
hipBLASLt.

## Minimal change

Keep the readable implementation and registry override. Change only `Auto` selection:

```text
ordinary GEMM:
  keep the old substantial-reduction rule

transpose(left) weight-gradient GEMM:
  if output rows and columns are both at least 128, use hipBLASLt
  even when the reduction is only 3 or 32
```

This rule corresponds to a wide weight-gradient matrix, not every small-reduction GEMM.
The public selector overload exposes transpose flags so tests and future offline tuning can
inspect or override the exact logical `(M,K,N)` key.

## Operator evidence

| Logical M×K×N | readable | hipBLASLt | Speedup | Max error |
|---|---:|---:|---:|---:|
| 896×3×896 | 0.252 ms | 0.164 ms | 1.54× | 3.0e-8 |
| 896×3×4864 | 0.667 ms | 0.169 ms | 3.95× | 6.0e-8 |
| 4864×3×896 | 0.668 ms | 0.171 ms | 3.92× | 3.0e-8 |
| 896×32×896 | 0.857 ms | 0.174 ms | 4.94× | 1.2e-7 |
| 896×32×4864 | 3.644 ms | 0.181 ms | 20.09× | 2.4e-7 |
| 4864×32×896 | 3.646 ms | 0.166 ms | 21.99× | 2.4e-7 |

The full context-32 trace drops from 1.621 seconds to 0.382 seconds of Kernel time. The
507-call readable transpose hotspot is absent after the change. AdamW remains about
127–129 ms in both traces, confirming it was not the context-dependent cause.

## Official three-process result

| Shape B×T | Before | After | Self speedup | After / PyTorch | Peak change |
|---|---:|---:|---:|---:|---:|
| 1×3 | 18.79 | 31.17 tok/s | 1.659× | 0.734× | unchanged |
| 2×3 | 30.39 | 61.37 tok/s | 2.020× | 0.661× | unchanged |
| 1×32 | 60.02 | 268.63 tok/s | 4.476× | 0.547× | unchanged |
| 1×128 | 654.78 | 659.23 tok/s | 1.007× | 0.360× | unchanged |

![Weight-gradient routing result](../assets/bf16-weight-gradient-routing.svg)

All 24 candidate raw rows pass the same finite-loss, parameter-update, trained-token and
zero optimizer-payload-transfer gates. The largest numerical difference seen in the
operator matrix is `2.4e-7`. Full CPU/HIP, sanitizer and PyTorch-enabled suites pass.

## Decision and remaining failure

Keep. All four shapes improve, context 128 stays within the no-regression gate, and peak
memory is unchanged.

This does not reach PyTorch parity: the best after-ratio is 0.734× and context 128 remains
0.360×. After the context-32 hotspot disappears, AdamW, ordinary readable matmul, strided
copy, cast and reductions become visible. The next profile must use the retained route and
choose one of those categories; it must not keep lowering a global threshold without exact
shape evidence.
