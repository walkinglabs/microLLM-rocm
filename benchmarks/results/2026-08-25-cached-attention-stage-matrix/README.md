# Cached Attention stage matrix

This directory is the immutable MI300X evidence for Experiment 282.

The measured binary came from commit `b496518a02cb0ba1b5472ff84ca2006b4fca8b15` with a clean
worktree before the output directory was created. The machine reported AMD Instinct MI300X VF,
`gfx942`, 304 compute units, ROCm 7.13.0, CMake 3.31.10 and GCC/G++ 13.3.0.

Reproduce the 24 fresh-process rows with:

```bash
cmake --preset hip-release
cmake --build --preset hip-release --target microllm_bench_cached_attention_stages
python3 benchmarks/single_gpu/cached_attention_stage_matrix.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_cached_attention_stages \
  --output-directory benchmarks/results/2026-08-25-cached-attention-stage-matrix \
  --sequences 512,2048 --batches 1,2 --cache-dtypes fp32,bf16 \
  --runs 3 --warmup 3 --repetitions 20
```

Files:

- `raw.jsonl`: all 24 process records, including complete-output errors and allocation/transfer counters;
- `summary.json`: eight case medians, process ranges and stage shares;
- `analysis.json`: bounded interpretation and the next falsifiable hypothesis;
- `verification.json`: exact evidence gate used by the optimization-log validator;
- `stage-timing.svg`: generated autoresearch-style visual; `raw.jsonl` remains authoritative.

The transparent three-stage pipeline is deliberately not the production implementation. Its generic
softmax share identifies where that readable decomposition spends time, but it does not prove that the
same percentage belongs to the softmax instructions inside the much faster fused Kernel.
