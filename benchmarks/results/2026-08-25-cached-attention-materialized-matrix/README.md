# Exact-order materialized-score Attention matrix

This directory is the clean MI300X evidence for Experiment 285. The measured
benchmark and runner came from commit `6315385fbe2454b93da2343288f8a1a7e3f0ff1c`.

```bash
python3 benchmarks/single_gpu/cached_attention_materialized_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory benchmarks/results/2026-08-25-cached-attention-materialized-matrix \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --runs 3 --warmup 3 --repetitions 20
```

All 24 process records are in `raw.jsonl`; `summary.json` contains eight medians;
`analysis.json` and `verification.json` lock the bounded conclusion; and
`comparison.svg` is generated from the summary. Every complete context is bitwise
equal to current fused Attention.
