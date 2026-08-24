# Experiment 203 — BTHD across sequence and batch

Status: keep explicit measured cases

## Result

| Model | Case | Speedup | Peak saved | Attention copies | Residual copies |
|---|---|---:|---:|---:|---:|
| Qwen | B1/T256 | 1.1421× | 2 MiB | 0 | 0 |
| Qwen | B1/T1024 | 1.0989× | 8 MiB | 0 | 0 |
| Qwen | B2/T512 | 1.0852× | 8 MiB | 0 | 1 / 7,168 B |
| DeepSeek | B1/T256 | 1.0947× | 3.5 MiB | 0 | 0 |
| DeepSeek | B1/T1024 | 1.0914× | 14 MiB | 0 | 0 |
| DeepSeek | B2/T512 | 1.0923× | 14 MiB | 0 | 1 / 12,288 B |

![BTHD sequence and batch](../assets/inference-bthd-shape-models.svg)

All 36 performance outputs are bit-exact with per-row top-1 equality. Six separate diagnostics
prove Attention source copies are zero.

## Diagnostic boundary correction

The first diagnostic predicate incorrectly required total strided calls to be zero. B2 retains one
unspecified last-row selection copy outside Attention. The gate now requires attention.layout/core
to be zero and reports residual traffic separately.

## Decision

Keep explicit BTHD for these six cases. Do not broaden to cached-prefill or value-trace paths;
their layout contracts remain unmeasured.

Raw evidence:
[benchmarks/results/2026-08-24-inference-bthd-shape-models/](../../../benchmarks/results/2026-08-24-inference-bthd-shape-models/).
