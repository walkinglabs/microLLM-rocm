# Public online-Attention operator matrix

This directory validates the public `online_causal_gqa_attention_bthd` API after the
benchmark-only prototype was moved into `microllm::ops`.

```bash
cmake --preset hip-release
cmake --build build/hip-release \
  --target microllm_bench_online_attention_operator --parallel
ROCR_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/rocwmma_online_operator_matrix.py \
  --binary build/hip-release/benchmarks/microllm_bench_online_attention_operator \
  --output-directory benchmarks/results/2026-08-25-rocwmma-online-operator \
  --runs 3 --warmup 5 --repetitions 20
```

The 14-case matrix includes native Qwen/DeepSeek B1/B2 T32/T512 and B1 T1024,
plus T31/T33 and D32 fallbacks. Three fresh processes per case produce 42 raw
records. Native cases report exactly 25 native and zero fallback calls; fallback
cases report the reverse. Timed payload transfers are zero.

All ten native cases improve over the current operator by 1.534×–2.456× and write
no global score Tensor. The four fallbacks remain numerically exact but reach only
0.607×–0.696× because three explicit BF16→FP32 casts are part of the public contract.
The result admits a model-level experiment; it does not enable a model route.
