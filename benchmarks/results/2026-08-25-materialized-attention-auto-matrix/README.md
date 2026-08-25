# Scoped automatic materialized-score policy verification

Experiment 288 verifies the user-visible default from clean commit
`a3abca0d8a3ff8822cf0c29325ffc0369f5e6d08`. Current processes explicitly pass
`--cached-attention-materialized false`; candidate processes omit the option and must
report `auto-enabled`.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/materialized_attention_model_matrix.py \
  --comparison-runner benchmarks/single_gpu/compare_cached_attention_split_models.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-materialized-attention-auto-matrix \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --contexts 2048 --batches 1,2 --decode-tokens 32 --cache-dtype bf16 \
  --candidate-policy auto --minimum-sequence 2048 \
  --warmup 1 --steps 3 --runs 3
```

Four child directories retain 24 process rows, 12 pairs, full logits/tokens and charts.
Root files aggregate and validate the scoped automatic policy.
