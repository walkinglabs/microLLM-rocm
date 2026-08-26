# First block-0 drift after exact Q/K/V cache

This trace runs DeepSeek full prefill with scoped Q solution 296100 and K/V
solution 292135. It captures the first two batch rows through block 0, from
embedding to final block output, for B1/B2/B4/B8 in two fresh processes.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_post_cache_block0_trace.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-26-post-cache-block0-trace \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 --runs 2
```

![Post-cache block-0 trace](post-cache-trace.svg)

Q/K/V projections, RoPE, current value, and BF16 cache key/value are bitwise
equal across every batch. The first renewed difference is Attention context:
Max is `9.77516e-6` at B2 and `0.000332177` at B4/B8. Context rows inside the
same batch remain bitwise equal.

Attention output projection then introduces within-batch row differences and
passes them through residual, FFN, and block output. At B8, Attention output Max
is `0.000163078`, FFN output Max is `0.000148535`, and block output Max is
`0.000263214`.

All metrics repeat across both fresh processes. The next experiment decomposes
the full-prefill Attention core into QK scores, causal softmax, and P×V before
selecting a solution or kernel change. The invariant-QKV route remains explicit
and default-off.

`raw.jsonl` and `summary.json` are authoritative.
