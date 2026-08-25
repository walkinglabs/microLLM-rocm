# DeepSeek exact-softmax split-P*V model gate

The current side explicitly enables the retained materialized exact-order route.
The candidate explicitly disables materialized Attention and enables split-P*V
S16. Three fresh pairs alternate process order.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/compare_cached_attention_split_models.py \
  --manifest /path/to/pinned-deepseek-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-25-cached-attention-split-pv-model \
  --model deepseek-r1-distill-qwen-1.5b \
  --candidate-policy split-pv --context 2048 --batch 2 \
  --decode-tokens 64 --cache-dtype bf16 --splits 16 \
  --minimum-sequence 2048 --warmup 2 --steps 5 --runs 3 \
  --maximum-logit-error 0.001 --maximum-logit-rms 0.0001
```

![Split-P*V model comparison](comparison.svg)

Performance passes: median throughput rises from 177.52 to 263.20 tokens/s,
or 1.4834x, and every leave-one-pair-out value is at least 1.4829x. All 64
generated tokens match, peak memory is unchanged, and KV cache remains
121,110,528 bytes.

Precision fails. Every pair reports the same complete-logit Max/RMS difference,
0.064486/0.011488, over 303,872 values. This is far above the 0.001/0.0001
gate. The route is rejected despite the speed and matching top-1 tokens. No Qwen
boundary matrix or automatic policy is admitted.

The candidate adds 17,920 logical allocation calls and 65 cold backend calls in
the five-generation process while leaving peak unchanged. `raw.jsonl`,
`pairs.jsonl`, and `summary.json` are authoritative; `analysis.json` records the
decision and `verification.json` pins the clean commit and full gates.
