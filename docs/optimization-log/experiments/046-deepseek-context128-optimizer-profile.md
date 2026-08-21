# Experiment 046 — DeepSeek context 128 optimizer profile

## Question

After wide weight-gradient GEMM routing and fused causal GQA, which category is now the
largest trustworthy training hotspot on the 1.5B model?

## Contract

- Source: `1d047b9`.
- Device: AMD Instinct MI300X VF (`gfx942`).
- Model: DeepSeek-R1-Distill-Qwen-1.5B.
- Shape: batch 1, context 128.
- Precision: BF16 Linear mirrors with FP32 masters.
- Window: checkpoint loading, one warm-up step, two measured steps.
- Tool: rocprofv3 runtime trace, aggregated by `scripts/summarize_rocprof.py`.

The scope is deliberately written before interpretation. Because the trace starts at
process launch, loading kernels and training kernels coexist in the same CSV.

## Result

The trace contains 7,890 Kernel dispatches and 1.369 seconds of aggregate device Kernel
time. AdamW master/mirror update is the largest category:

| Category | Calls | Kernel time | Share | Safe attribution |
|---|---:|---:|---:|---|
| AdamW master + BF16 mirror | 1,017 | 450.798 ms | 32.94% | training |
| strided copy | 1,037 | 314.873 ms | 23.00% | loading + training mixed |
| fused causal GQA backward | 84 | 183.577 ms | 13.41% | training |
| FP32→BF16 cast | 788 | 82.204 ms | 6.01% | loading + training mixed |
| fill | 852 | 64.438 ms | 4.71% | loading + training mixed |
| fused causal GQA forward | 84 | 58.025 ms | 4.24% | training |
| RMSNorm weight gradient | 171 | 42.841 ms | 3.13% | training |
| bias gradient | 252 | 41.593 ms | 3.04% | training |
| all remaining kernels | 3,605 | 130.374 ms | 9.53% | mixed |

![DeepSeek context-128 bottleneck profile](../assets/deepseek-context128-profile.svg)

Two count identities make the AdamW and Attention attribution stronger than a name-only
guess:

```text
339 parameter tensors × 3 optimizer steps = 1,017 AdamW launches
28 Transformer layers × 3 steps = 84 Attention forward + 84 backward launches
```

`strided_copy` does not have such a clean training boundary. The new device-native loader
transposes Linear weights on the GPU, so a large fraction of those calls can be setup work.
Calling the whole 23% a backward bottleneck would be unsupported.

## Interpretation

The next experiment should reduce optimizer dispatches. A direct pointer-array AdamW
Kernel is unsafe today because `Value::zero_grad()` drops the gradient Tensor and the next
backward can produce a different allocation. A persistent device pointer table would then
contain stale addresses.

The next two nodes are therefore ordered:

1. retain correctly sized gradient buffers across `zero_grad`, while preserving the
   public meaning of “no current gradient”;
2. build one persistent pointer table and update many parameters in a small number of
   multi-tensor AdamW launches.

## Falsification

Stable buffers are not a performance result by themselves. If a multi-tensor optimizer
reduces launches but the formal Qwen/DeepSeek shape medians do not improve by at least 5%,
the candidate is discarded and this profile remains only a map of where time was spent.
