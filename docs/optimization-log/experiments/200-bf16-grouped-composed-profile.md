# Experiment 200 — profile after both grouped policies

Status: measurement; grouped submission track locally saturated

## Phase delta

| Model | Baseline calls | Composed calls | GEMM speedup | Total speedup |
|---|---:|---:|---:|---:|
| Qwen | 217 | 145 | 1.182× | 1.009× |
| DeepSeek | 253 | 169 | 1.099× | 1.034× |

The 72/84 saved submissions equal three per block: two from QKV and one from gate/up.

![Post-composition profile](../assets/bf16-grouped-composed-profile.svg)

## What remains

GEMM remains 46.8%/59.1% of Kernel time, but independent equal-input projection pairs are now
exhausted. Cast plus strided materialization is 18.9%/14.8%. Causal softmax is another visible
per-block boundary.

The next valid candidate must fuse or change a larger Attention/cast/layout region and re-run
complete models. Another exact index or stateless GroupedGemm wrapper is already contradicted by
Experiments 189, 194 and 195.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-grouped-composed-profile/](../../../benchmarks/results/2026-08-24-bf16-grouped-composed-profile/).
