# 2026-08-19 — M5 reproducible micro-benchmark harness

## Contract

Separate device kernel time from synchronized wall time. Require warm-up, repeated
measurements, device/software metadata, and CPU-reference correctness in every emitted
record. Never substitute a fastest single observation for an end-to-end conclusion.

## Harness

`microllm_bench_ops` covers add, readable matmul, and Softmax on CPU or HIP. It emits
one schema-versioned JSON object containing:

- engine, device name, architecture, HIP runtime/driver, dtype and size;
- warm-up and repetition counts;
- kernel/Event and synchronized wall min/mean/max;
- maximum absolute error and output value sum;
- before/after free memory and total device memory.

## First committed 64x64 readable matmul smoke

| path | kernel/wall mean | min | max | max error |
|---|---:|---:|---:|---:|
| CPU | 0.142255 ms | 0.140625 | 0.151552 | 0 |
| gfx942 HIP kernel | 0.048241 ms | 0.028304 | 0.081665 | 2.38419e-07 |
| gfx942 synchronized wall | 0.059908 ms | 0.038648 | 0.090762 | 2.38419e-07 |

This tiny shape favors launch-sensitive behavior and does not establish GEMM
competitiveness. Before/after free memory changed during runtime initialization; it is
not peak memory. The raw JSONL is committed under `benchmarks/results`.

## Next experiment

Profile larger Model-S projection shapes, compare readable matmul with hipBLASLt, and
check whether operator improvement changes complete train/inference wall time.
