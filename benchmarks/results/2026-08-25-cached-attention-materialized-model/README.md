# Exact-order materialized-score official model gate

Experiment 286 uses the clean runner and binary from
`b62d302b3ae7af563eea1d851ff227e5439d1e25` on MI300X/gfx942 with ROCm 7.13.0.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/compare_cached_attention_split_models.py \
  --manifest /path/to/pinned-deepseek-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-cached-attention-materialized-model \
  --model deepseek-r1-distill-qwen-1.5b --candidate-policy materialized \
  --context 2048 --batch 2 --decode-tokens 64 --cache-dtype bf16 \
  --minimum-sequence 512 --warmup 2 --steps 5 --runs 3 \
  --maximum-logit-error 0 --maximum-logit-rms 0
```

`raw.jsonl` has six fresh processes, `pairs.jsonl` has three alternating pairs,
`summary.json` is the aggregate, `analysis.json` and `verification.json` lock the
decision, and `comparison.svg` is generated. The candidate remains explicit until a
broader model/shape gate establishes an automatic policy boundary.
