# Experiment 194 — a faster exact GEMM does not warm the process

Status: discard

## Question

Experiment 193 showed that loading every hipBLASLt kernel is much too broad. Can we instead select
one exact BF16-output gate/up solution that the model really uses, reduce cold start, and keep its
steady operator gain?

The gate requires both official models to:

- pass complete-logit correctness and peak-memory checks;
- improve cold first-forward latency by at least 1.02×;
- improve steady T512 throughput by at least 1.01×.

## Correctness-before-timing selection

Each shape is screened in three fresh tuner processes. All 64 candidates pass complete output in
all three runs. Selection uses the lowest median Event time among the common passing set.

| Model | BF16 output shape | Index | Operator speedup |
|---|---|---:|---:|
| Qwen | 512×896×4864 | 76074 | 1.059× |
| DeepSeek | 512×1536×8960 | 76091 | 1.032× |

Indices are valid only for the recorded backend environment. Registration still verifies exact
shape and support before dispatch.

## Complete-model result

The 24 model processes rotate phase and policy order.

| Model | Default cold | Exact cold | Cold ratio | Default steady | Exact steady | Steady ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 4888.8 ms | 4935.9 ms | 0.990× | 93617 tok/s | 91088 tok/s | 0.973× |
| DeepSeek | 4907.3 ms | 4924.9 ms | 0.996× | 49955 tok/s | 50316 tok/s | 1.007× |

Process-wall ratios are 0.978×/0.981×. All logits are bit-exact and both peak ratios are 1.0.

![Exact BF16 startup gate](../assets/bf16-exact-startup.svg)

## Decision

Reject exact gate/up registration. The operator really is faster, but it neither shortens library
first use nor passes the two-model steady gate. This closes the one-shape exact-registration
shortcut. A future selected-kernel startup attempt needs an actual library/module loading API or
a persistent serving process, not another solution index on the first GEMM.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-exact-startup/](../../../benchmarks/results/2026-08-24-bf16-exact-startup/).
