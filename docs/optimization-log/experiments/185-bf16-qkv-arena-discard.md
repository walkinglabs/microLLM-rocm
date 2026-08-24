# Experiment 185 — BF16 QKV Arena does not move the model

Status: `discard` model policy; keep caller-owned primitive and opt-in diagnostic

## Hypothesis

After selective FFN removes long-prefill temporaries, the next repeated allocation is one shared
BF16 input cast plus Q/K/V outputs. Can caller-owned QKV storage add another two-model gain?

## Implementation boundary

`Bf16QkvWorkspace` contains input BF16 and three shape-specific BF16 fallbacks.
`bf16_qkv_projection_out_` writes caller FP32 Q/K/V and rejects dtype/shape/device/layout/alias
violations. Model cache entries additionally own three FP32 outputs in one backing allocation.

The formal baseline already enables the retained FFN `minimum_rows=512`. Candidate adds QKV
`minimum_rows=512`; therefore every difference is incremental QKV behavior.

## Formal matrix

| Model | Case | QKV / baseline | QKV bytes | Allocation calls |
|---|---|---:|---:|---:|
| Qwen | T32 B1 | 1.001× | 0 | 3135→3135 |
| Qwen | T512 B1 | 1.004× | 4.46 MB | 2895→2415 |
| Qwen | T32 B4 | 1.001× | 0 | 3020→3020 |
| Qwen | decode B1 | 0.976× | 0 | 10630→10630 |
| Qwen | decode B4 | 0.997× | 0 | 10635→10635 |
| DeepSeek | T32 B1 | 1.001× | 0 | 3515→3515 |
| DeepSeek | T512 B1 | 1.005× | 7.86 MB | 3375→2815 |
| DeepSeek | T32 B4 | 0.998× | 0 | 3520→3520 |
| DeepSeek | decode B1 | 1.009× | 0 | 23650→23650 |
| DeepSeek | decode B4 | 0.999× | 0 | 23515→23515 |

All 60 complete logits and fixed decode tokens pass. Eligible QKV rows have one entry; all short
rows have zero entries/capacity/eligible calls. Neither eligible row reaches 1.01.

![BF16 QKV Arena discard](../assets/bf16-qkv-arena-discard.svg)

## Profiler and falsification

Qwen T512 keeps 5,642 Kernels, 4,000 `hipLaunchKernel` and 1,519 extended launches. malloc/free
falls 1,637/1,327→1,446/1,135 and Kernel time is 49.63/49.27 ms. The hypothesis “removing the
next allocation family materially improves end-to-end prefill” is falsified: Attention/GEMM math
now dominates enough that the remaining host allocation reduction buys only 0.4%–0.5%.

## Decision

Do not enable QKV Arena as a model policy. Retain the out primitive, tests and explicit default-off
diagnostic so the evidence remains reproducible and future Graph/liveness work has a safe seam.
The next optimization must start from a new profile/allocation-size attribution rather than moving
another guessed Tensor family into persistent storage.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-qkv-arena-discard/`](../../../benchmarks/results/2026-08-24-bf16-qkv-arena-discard/).
