# DeepSeek cached step-0 block drift

This diagnostic compares batch 1 with row 0 of batch 2 for the same 2048-token
prompt. It runs two fresh processes for FP32 Linear and two for BF16 FFN-only,
then compares the complete output of the embedding, all 28 Transformer blocks,
the final normalization, and all 151,936 logits.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_cached_block_drift.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-25-deepseek-cached-block-drift \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 \
  --runs 2 --trace-max-elements 400000
```

![Cached block drift](block-drift.svg)

The embedding is bitwise equal across batch shapes. The first difference appears
at block 0. Its FP32 Linear Max/RMS error is `7.62194e-6/1.91177e-6`; BF16
FFN-only reaches `0.00390911/0.000348230`, a 512.88-fold Max amplification at
that boundary. The BF16 error grows to `0.582840/0.0545060` at block 27 and ends
at `0.0629854/0.0251711` in the logits after final normalization.

This is a location result, not a claim that block 0 alone causes the final token
change. It does not change the precision or scheduler defaults. The next
experiment opens block 0 and compares its attention residual, FFN input cast,
gate/up activations, down projection, and block output.

`raw.jsonl` and `summary.json` are authoritative. `analysis.json` records the
interpretation; `verification.json` pins the clean measurement commit and gates.
