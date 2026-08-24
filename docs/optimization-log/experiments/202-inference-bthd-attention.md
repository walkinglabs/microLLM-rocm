# Experiment 202 — inference BTHD Attention island

Status: keep explicit policy; default unchanged

## Design

No new math Kernel is required. Existing rope_split_half_bias_bthd reads contiguous projection
order and writes Q/K in Attention order. Existing causal_gqa_attention_bthd consumes V in
projection order and returns context in output-linear order.

The model route is deliberately narrow: HIP, T≥256, BF16 Attention, split-half+bias Q/K,
no prefill-cache write and no value trace. The old readable path remains the fallback.

## Formal result

| Model | Speedup | Calls | Bytes | Peak saved | Max/RMS |
|---|---:|---:|---:|---:|---:|
| Qwen | 1.1146× | 96→0 | 100.7 MB removed | 4.0 MiB | 0/0 |
| DeepSeek | 1.0936× | 112→0 | 205.5 MB removed | 7.0 MiB | 0/0 |

![Inference BTHD Attention](../assets/inference-bthd-attention.svg)

Twelve performance processes run without diagnostics; twelve separate zero-warmup processes prove
copy elimination. Formal complete logits are bit-exact.

## Decision

Keep the explicit policy and CLI flag. Do not enable unsupported RoPE/bias/cache/value-trace
combinations. The next matrix should test rows256/1024 and batch2 before considering a broader
selection rule.

Raw evidence:
[benchmarks/results/2026-08-24-inference-bthd-attention/](../../../benchmarks/results/2026-08-24-inference-bthd-attention/).
