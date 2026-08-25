# Exact-order finalize physical-thread mapping matrix

This matrix asks one narrow question: can 64 or 128 physical threads schedule
the existing logical 256-lane finalizer faster without changing any floating-point
operation order?

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/cached_attention_finalize_mapping_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory \
    benchmarks/results/2026-08-25-cached-attention-finalize-mapping \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --finalize-threads 64,128,256 --runs 2 --warmup 3 --repetitions 20
```

![Finalize mapping matrix](mapping.svg)

All 96 fresh processes pass complete-context bitwise equality and warm backend
allocation remains zero. No case passes the Event 1.05x plus wall 1.02x performance
gate. The 128-thread mapping ranges from 0.9901x to 1.0121x in Event time and
0.9808x to 1.0121x in wall time versus 256 threads. The 64-thread mapping ranges
from 0.5548x to 0.9651x in Event time.

The target DeepSeek T2048/B2/BF16 case reaches only 1.0121x in both Event and wall
time, so there is no official-model experiment and no default-policy change. The
explicit research overload remains useful for reproducing this hardware result.
The next experiment retains exact scores and exact softmax and changes only the
P*V accumulation architecture.

`raw.jsonl` and `summary.json` are authoritative. `analysis.json` records the
decision and `verification.json` pins the clean commit and full gate results.
