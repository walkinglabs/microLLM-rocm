# DeepSeek microLLM cross-batch complete-logit audit

This audit compares microLLM with itself. B1 is the reference; B2/B4/B8 receive
identical prompt rows. Decode steps 0, 1, and 2 each get two fresh processes.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_cached_cross_batch_logits.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-25-deepseek-cross-batch-logits \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 \
  --batches 1,2,4,8 --decode-steps 0,1,2 --runs 2 \
  --warmup 1 --cache-dtype bf16
```

![Cross-batch complete logits](cross-batch.svg)

All 24 processes are deterministic. Every row within B2/B4/B8 is bitwise equal,
and host argmax of all 151,936 logits agrees with the device-selected token.
This rules out row indexing/storage corruption and device argmax as explanations.

Cross-batch complete logits already differ at step 0. Maximum Max/RMS difference
over the audit is 0.197803/0.046133. At step 2, B1/B8 select token 151643 while
B2/B4 select 3555. This exactly explains the batch-dependent sequence observed in
Experiment 294.

The next experiment isolates the batch-shape numerical path with FP32,
BF16-FFN-only, BF16-Attention-only, and combined BF16 at step 0. No scheduler
default is admitted. `raw.jsonl` and `summary.json` are authoritative;
`analysis.json` records the inference and `verification.json` pins full gates.
