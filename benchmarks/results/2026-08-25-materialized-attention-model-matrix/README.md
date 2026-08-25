# Materialized-score official model boundary matrix

Experiment 287 was measured from clean commit
`8eebab67423bfaec7db079cd5db72d86a4989d17` on MI300X/gfx942, ROCm 7.13.0.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/materialized_attention_model_matrix.py \
  --comparison-runner benchmarks/single_gpu/compare_cached_attention_split_models.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-materialized-attention-model-matrix \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --contexts 512,2048 --batches 1,2 --decode-tokens 32 \
  --cache-dtype bf16 --minimum-sequence 512 \
  --warmup 1 --steps 3 --runs 3
```

Each of the eight child directories keeps six process rows, three pairs, full logits,
tokens and its own chart. Root `cases.jsonl`, `summary.json`, `analysis.json`,
`verification.json`, and `matrix.svg` establish the cross-model boundary.
