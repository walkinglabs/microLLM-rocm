# Experiment 020 — offline exact-shape hipBLASLt solution

Status: `discard`

## Hypothesis

DeepSeek's trace is dominated by M=1 GEMMs. An official `hipblaslt-bench
--algo_method all` search for the physical `1536×1×1536` problem initially selected
solution 293437 at 8.40 us. Passing that exact algorithm for user shape
`1×1536×1536` might outperform `algo=null`.

## Safety boundary

The candidate was restricted to:

- gfx942;
- hipBLASLt version 100300;
- FP32 user shape `M=1,K=1536,N=1536`;
- NN layout and zero workspace;
- successful `getAlgosFromIndex` and `matmulIsAlgoSupported`;
- every other case falling back to `algo=null`.

This matters because AMD explicitly documents that solution indices cannot be reused
across library versions.

## Operator result

The framework profile confirmed the explicit `GRVWA1...` Kernel was selected and the
FP32 maximum error was `2.74e-6`. However, the longer official-bench repeat narrowed the
apparent gain:

```text
explicit solution 293437      9.50 us
default heuristic 293832      9.87 us
difference                     3.7%
```

The initial 8.40 us all-search winner was therefore not a stable 3× operator gain.

## Official-model result

```text
DeepSeek candidate samples    56.74 / 55.67 / 56.39 token/s
candidate median              56.39 token/s
retained median               58.32 token/s
change                        -3.3%
candidate score               1.829748
retained score                1.845199
```

All eight generated tokens remained exact. Qwen and training shapes were unchanged.

## Decision

`discard`. The explicit algorithm is numerically correct and locally measurable, but
the stable micro difference is small and the official model regresses. The version- and
shape-specific code is removed. This is the requested falsification case: a faster
selected GEMM does not establish an end-to-end optimization.

Raw evidence is in [020-data](020-data/README.md).
