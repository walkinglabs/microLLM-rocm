# rocWMMA online causal-GQA prototype

This directory records a benchmark-only online Attention prototype. It is not yet a
public operator or model route.

## Reproduce

```bash
cmake --preset hip-release
cmake --build build/hip-release \
  --target microllm_bench_rocwmma_online_attention --parallel
ROCR_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/rocwmma_online_attention_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_rocwmma_online_attention \
  --output-directory benchmarks/results/2026-08-25-rocwmma-online-attention \
  --runs 3 --warmup 5 --repetitions 20
```

Environment: AMD Instinct MI300X VF (`gfx942:sramecc+:xnack-`), ROCm 7.13,
rocWMMA 2.2.0. The matrix uses Qwen-style H14/KV2/D64 and DeepSeek-style
H12/KV2/D128 GQA at T32–2048. Every one of 14 shapes runs in three fresh
processes and compares all output elements with a CPU BF16-input reference.

The candidate uses 32×32 rocWMMA QK, online max/sum, BF16 probability tiles and
rocWMMA PV with FP32 accumulation. It never writes a global score tensor. The matched
current-framework baseline calls `causal_gqa_attention_bthd` on the same rounded
values represented as FP32.

All candidate/current Event ratios are at least 1.260×. At T2048 it avoids
234,881,024 Qwen score bytes and 201,326,592 DeepSeek score bytes. Short scalar fused
kernels remain faster, and batch/tail support is absent, so the result only admits
public-operator integration with an explicit fallback.
