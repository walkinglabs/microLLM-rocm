# Exact-softmax split-P*V cached-Attention matrix

This matrix keeps parallel scores and the exact current 256-lane softmax order.
Only the final probability-times-value accumulation is split into contiguous
sequence ranges.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/cached_attention_split_pv_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory \
    benchmarks/results/2026-08-25-cached-attention-split-pv-matrix \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --splits 1,2,4,8,16 --runs 2 --warmup 3 --repetitions 20
```

![Split P*V search](split-pv-search.svg)

All 160 fresh processes pass complete-context accuracy. S1 is bitwise equal to
the exact-order materialized current path and is slower in all 16 cases, proving
that extra buffers and launches alone do not explain a win. S16 wins every case.

Winner Event speedup ranges from 1.2749x to 2.9549x and synchronized wall speedup
from 1.2372x to 2.6373x. Maximum context Max/RMS error is
3.90e-9/1.09e-9. The target DeepSeek T2048/B2/BF16 case reaches
2.2908x Event and 2.1372x wall speed with 196,608 bytes each for probabilities
and partials.

This admits one explicit official-model gate; it does not retain a model route or
change Auto. `raw.jsonl` and `summary.json` are authoritative. `analysis.json`
records the decision, while `verification.json` pins the clean commit and full
test gates.
