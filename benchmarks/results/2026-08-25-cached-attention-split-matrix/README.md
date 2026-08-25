# Cached Attention split-sequence search

This directory is the immutable MI300X operator evidence for Experiment 283.

The measured binary and runner came from clean commit
`eace0ce0035b9e1191f11da040ed709b40fbe7f7`. The machine reported AMD Instinct
MI300X VF, `gfx942`, 304 compute units, ROCm 7.13.0, CMake 3.31.10 and GCC/G++
13.3.0.

Reproduce all 144 fresh-process records with:

```bash
cmake --preset hip-release
cmake --build --preset hip-release --target microllm_bench_cached_attention_stages
python3 benchmarks/single_gpu/cached_attention_split_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory benchmarks/results/2026-08-25-cached-attention-split-matrix \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --splits 1,2,4,8,16,32 --runs 3 --warmup 3 --repetitions 20
```

- `raw.jsonl` contains every process measurement and complete-output/resource gate.
- `summary.json` contains 48 candidate medians and eight shape winners.
- `analysis.json` separates the operator decision from the still-unproven model decision.
- `verification.json` is consumed by the optimization-log validator.
- `split-search.svg` is generated from the summary; raw JSONL remains authoritative.

The public candidate is explicit and the ordinary model route is unchanged at this
measurement commit.
