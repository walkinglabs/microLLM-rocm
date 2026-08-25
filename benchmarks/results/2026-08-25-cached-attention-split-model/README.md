# Split-sequence official DeepSeek model gate

This directory records Experiment 284. The runner and model binary were taken from
clean commit `af75f378e56aed3371b34e5a3e320730d6279895`. Hardware and software were
AMD Instinct MI300X VF (`gfx942`), ROCm 7.13.0, CMake 3.31.10 and GCC/G++ 13.3.0.

Reproduce with:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/compare_cached_attention_split_models.py \
  --manifest /path/to/pinned-deepseek-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-cached-attention-split-model \
  --model deepseek-r1-distill-qwen-1.5b \
  --context 2048 --batch 2 --decode-tokens 64 --cache-dtype bf16 \
  --splits 32 --minimum-sequence 512 \
  --warmup 2 --steps 5 --runs 3 \
  --maximum-logit-error 0.001 --maximum-logit-rms 0.0001
```

The performance gate passes and all 64 generated tokens match, but the complete-logit
gate fails. `status: failed` is therefore the expected evidence result, not a broken
runner. The explicit research route remains default-off.

Files are six process records in `raw.jsonl`, three paired records in `pairs.jsonl`,
their `summary.json`, bounded interpretation in `analysis.json`, an exact
`verification.json`, and the generated `comparison.svg`.
