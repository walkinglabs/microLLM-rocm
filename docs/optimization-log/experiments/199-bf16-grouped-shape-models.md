# Experiment 199 — equal GEMM rows do not mean equal workloads

Status: keep explicit rows256/1024 policies; default unchanged

## Matrix

| Model | Case | Rows | Speedup | Peak ratio | Max/RMS |
|---|---|---:|---:|---:|---:|
| Qwen | B1/T256 | 256 | 1.1075× | 1.00176 | 0.10463/0.02117 |
| Qwen | B1/T1024 | 1024 | 1.0280× | 1.00644 | 0.15857/0.03290 |
| Qwen | B2/T512 | 1024 | 1.0311× | 1.00657 | 0.12461/0.02470 |
| DeepSeek | B1/T256 | 256 | 1.0755× | 1.00088 | 0.05011/0.00870 |
| DeepSeek | B1/T1024 | 1024 | 1.0212× | 1.00338 | 0.03345/0.00731 |
| DeepSeek | B2/T512 | 1024 | 1.0223× | 1.00340 | 0.06598/0.00959 |

All 36 processes pass. B2 top-1 is checked independently per row. Both grouped registries dispatch
exactly once per block per forward.

![Grouped sequence/batch model matrix](../assets/bf16-grouped-shape-models.svg)

## Failure found before the result

The first formal run discovered that logits-output wrote only one vocabulary row even when
last-logits Tensor contained batch rows. The run stopped before comparison and is not evidence.

The CLI now writes all rows in last mode. In full mode it gathers each batch row's own final token.
A generated tiny Qwen-style checkpoint runs B1, B2-last and B2-full through the real binary;
B2 files contain exactly twice the values and both rows match B1.

## Decision

Keep explicit rows256/1024 composed policies. B1/T1024 and B2/T512 share projection keys but have
different speedups, so workload documentation must retain batch and sequence separately. No
version-local index becomes a default.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-grouped-shape-models/](../../../benchmarks/results/2026-08-24-bf16-grouped-shape-models/).
