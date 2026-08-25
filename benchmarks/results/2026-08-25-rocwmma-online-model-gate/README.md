# Full-model rocWMMA online-Attention gate

This directory compares the retained current BTHD BF16-Q/K path with the public
online-Attention operator inside pinned Qwen2.5-0.5B and
DeepSeek-R1-Distill-Qwen-1.5B models.

```bash
ROCR_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/rocwmma_online_model_matrix.py \
  --manifest /path/to/pinned-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-rocwmma-online-model-gate \
  --runs 3 --warmup 2 --steps 5
```

The 36 fresh processes cover B1T256, B1T1024 and B2T512 for both models and
both policies. Every online process reports exactly 168 Qwen or 196 DeepSeek
native calls and zero fallback calls. Each comparison uses all B×151,936 final
logits, top-token rows, prefill throughput and engine peak bytes.

The model route is rejected. All six throughput ratios regress to
0.761×–0.884×. Peak memory falls by 3.5–57.0 MiB and every top token stays equal,
but Qwen reaches Max/RMS logit drift 0.511/0.112. The public operator remains
available; model and CLI defaults remain unchanged.
