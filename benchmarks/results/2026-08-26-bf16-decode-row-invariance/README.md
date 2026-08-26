# BF16 gate/up operator row invariance

This matrix removes the Transformer and tests the 64 hipBLASLt solutions that
are common to DeepSeek gate/up shapes `M=1/2/4/8, K=1536, N=8960`. One
deterministic BF16 input row is repeated across M; every candidate is checked
against a complete CPU BF16 reference and across all output rows.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/bf16_row_invariance_matrix.py \
  --inventory-binary \
    build/hip-release/benchmarks/microllm_bench_bf16_algorithms \
  --row-invariance-binary \
    build/hip-release/benchmarks/microllm_bench_bf16_row_invariance \
  --output-directory \
    benchmarks/results/2026-08-26-bf16-decode-row-invariance \
  --warmup 1 --repetitions 3
```

![BF16 row invariance](row-invariance.svg)

All 64 candidates are supported, all 64 match the complete CPU BF16 reference
exactly, and all 64 produce bitwise-identical row 0 across M1/2/4/8. Maximum
reference and row-invariance error are both zero. Four candidates require no
workspace; candidate 75788 is the fastest zero-workspace sample in this run.

Candidate 75892 is also exact at operator level, confirming that its failure in
Experiment 300 is not intrinsic row dependence for identical BF16 gate input.
The full model feeds different BF16 values into the FFN because an upstream
difference already exists. No algorithm is promoted. The next experiment
exports block-0 BF16 K/V cache prefixes after prefill to determine whether the
first decode Attention-context drift was stored before decode or created by the
decode Attention kernel.

`inventory.json`, `raw.jsonl`, and `summary.json` are authoritative.
