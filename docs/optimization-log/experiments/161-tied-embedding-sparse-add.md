# Experiment 161 — sparse accumulation for tied embedding/output head

## Attribution before design

Default-off thread-local diagnostics record Autograd target operation/shape/source and
Runtime strided-copy layouts only in dedicated runs. Qwen T512 has 121 accumulation adds
over 191,184,896 elements. One tied `[151936,896]` leaf accounts for 136,134,656 elements
(`71.2%`). Its measured order is:

```text
first_source = matmul_right
last_add_source = embedding_backward
```

The old path therefore computes a dense output-head gradient, allocates and clears a
second 544 MB embedding gradient containing only 512 token rows, then adds all 136M
elements. DeepSeek is untied and has no matching leaf.

Both models' 240/280 strided copies collapse to four Attention transpose layouts; they
are recorded for a later experiment and are not mixed into this candidate.

## Candidate

When an embedding leaf already has a contiguous, shape-matching, uniquely owned dense
gradient, `embedding_backward_add_` atomically scatters token rows directly into it.
Otherwise the original dense `embedding_backward + add` remains. The destination
ownership check counts the node as the only persistent Storage owner.

CPU/HIP operator tests cover duplicate indices, caller Storage identity, zero payload
transfer and dense-reference equality. A tied/untied graph test compares the optimized
tied gradient against two independent weights and also runs the explicit dense fallback.

## Same-binary model A/B

`--tied-embedding-sparse-add true/false` changes only this path. Each model/policy uses
three fresh processes, BF16 Linear/FP32 masters, T512/B1, one warm-up and two measured steps.

| Model | Dense | Sparse | Throughput | Peak | Loss relative diff |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,290.77 | 14,547.74 tok/s | 1.018× | 13.025→11.969 GB (`-8.11%`) | 0.0207% |
| DeepSeek Distill 1.5B | 6,065.09 | 6,101.70 tok/s | 1.006× | unchanged | 0 |

The observed parameter guard is equal. Qwen allocation calls fall 3824→3820 across
three executed steps. DeepSeek reports zero sparse calls.

rocprofv3 preserves three embedding-backward launches but removes three dense add and
three dense fill launches: add 507→504, fill 586→583. Total Kernel time falls
116.41→113.20 ms in the profiled process.

![Tied embedding sparse add](../assets/tied-embedding-sparse-add.svg)

## Decision

Keep as a memory optimization. The declared gate is Qwen peak ratio ≤0.95, Qwen and
DeepSeek throughput ratio ≥0.98, final-loss relative difference ≤0.5%, equal parameter
guard, and no DeepSeek routing. All pass. This avoids repeating Experiment 010's generic
in-place hypothesis: the optimization is limited to a source-ordered, sparse second
contribution with a unique destination.

Raw evidence is in
[`benchmarks/results/2026-08-23-tied-embedding-sparse-add/`](../../../benchmarks/results/2026-08-23-tied-embedding-sparse-add/).
