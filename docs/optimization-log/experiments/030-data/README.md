# Experiment 030 raw evidence

These files belong to the BF16 FFN operator track. They do not change the FP32
`results.tsv` running best.

## Reproduce

```bash
cmake --preset hip-release
cmake --build --preset hip-release --parallel
python3 benchmarks/micro/run_bf16_ffn.py \
  --benchmark build/hip-release/benchmarks/microllm_bench_bf16_ffn \
  --raw-output docs/optimization-log/experiments/030-data/raw.jsonl \
  --summary-output docs/optimization-log/experiments/030-data/summary.json \
  --runs 3 --warmup 5 --repetitions 20
```

`raw.jsonl` contains all 36 independent-process results. `summary.json` groups the
three process medians for two model widths, two token counts and three execution paths.

The two kernel trace CSV files come from one structural run per path:

```bash
rocprofv3 -f csv --kernel-trace -- \
  build/hip-release/benchmarks/microllm_bench_bf16_ffn \
  --path per-linear --tokens 1 --hidden 896 --intermediate 4864 \
  --warmup 0 --repetitions 1
```

Repeat with `--path island`. Profiler timing is invalid for throughput comparison because
instrumentation and first-use library setup dominate. The CSVs are retained only to count
dispatches and check dtype transitions.

## Stable failure retained

Before commit `fd7de73`, direct BF16-input/FP32-output hipBLASLt GEMM returned status 6
for Qwen M values 1, 2, 8, 16 and 32; M=64 worked. The solution is a remembered per-shape
fallback to BF16 output plus a device cast, not a claim that every gfx942 shape has the
same threshold.
