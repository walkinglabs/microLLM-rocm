# DeepSeek cached block-0 internal drift

This diagnostic opens the first Transformer block selected by Experiment 297.
It compares batch 1 with row 0 of batch 2 for the same 2048-token prompt at
cached decode step 0. FP32 Linear and BF16 FFN-only each run in two fresh
processes.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_cached_block_detail.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-26-deepseek-cached-block-detail \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 \
  --runs 2 --trace-max-elements 400000
```

![Cached block-0 internal drift](block-detail.svg)

Attention norm, Q/K/V projections, RoPE, and current value are bitwise equal.
The materialized cached-Attention context introduces the first small difference:
Max/RMS/relative-L2 is `5.61606e-5/1.53427e-6/5.73506e-6`.

FFN norm remains at Max `2.98023e-6`. Converting that input to BF16 raises Max
to `0.000488281` and relative-L2 to `0.000101091`: 163.84x and 23.38x over the
preceding FP32 boundary. Gate is the first stage above Max `1e-3`, reaching
`0.0078125`; down projection reaches the block's largest relative-L2,
`0.00114281`.

Both fresh runs are identical, and the two rows inside B2 remain bitwise equal
through every selected block-0 boundary. This is diagnostic evidence, not a
performance measurement. No precision or scheduler default changes. The next
counterfactual keeps only block-0 FFN in FP32 and checks complete logits across
all 28 blocks before considering a broader policy.

`raw.jsonl` and `summary.json` are authoritative. `analysis.json` records the
interpretation; `verification.json` pins the clean measurement commit and gates.
