# DeepSeek block-0 prefill cache-prefix audit

This diagnostic exports raw BF16 K/V bytes after full prefill and before the
first decode step. It compares block 0 across B1/B2/B4/B8 for the same
2048-token prompt, using FP32 Linear weights and BF16 KV storage.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/audit_prefill_cache_prefix.py \
  --manifest /path/to/pinned-two-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-26-deepseek-prefill-cache-prefix \
  --model deepseek-r1-distill-qwen-1.5b --context 2048 --runs 2
```

![Block-0 prefill cache prefix](cache-prefix.svg)

The cache already differs before decode. Key reaches Max/RMS/relative-L2
`0.03125/8.65333e-5/9.03928e-7`; value reaches
`0.0009765625/1.53082e-5/4.53752e-5`. Every comparison covers all 524,288 BF16
values in one cache row.

Both fresh runs repeat bitwise. B2 rows are bitwise equal internally, while
B4/B8 rows are already different inside the same prefill batch. Therefore the
first decode Attention-context drift is consuming a batch-dependent stored
prefix; materialized decode Attention is not the first source.

No performance claim is made because cache export packs and copies diagnostic
bytes to the host. The next experiment traces block-0 full-prefill embedding,
attention norm, Q/K/V projections, RoPE, and BF16 cache-store boundaries.

`raw.jsonl` and `summary.json` are authoritative.
