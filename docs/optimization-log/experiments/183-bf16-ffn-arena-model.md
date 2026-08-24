# Experiment 183 — complete-model BF16 FFN Arena

Status: `keep` opt-in infrastructure; reject universal routing

## Hypothesis

Experiment 182 removed every FFN-region allocation, but an LLM has Attention, residuals,
KV cache and output projection around that region. Does one workspace shared across all
blocks improve complete inference without multiplying memory by layer count?

## Design

The model caches one backing allocation per exact `(device, flattened rows)` shape. Its
non-owning Tensor slices hold input cast, gate/up, activated, fallback and FP32 output.
Every block reuses the same entry on the legacy default Stream. This is safe because the
consumer residual add is enqueued before the next block overwrites the output.

The policy is explicit and default-off. Device moves clear the cache; disable clears memory
and statistics. Detailed value traces keep the allocation-returning diagnostic path. Concurrent
calls on one enabled model require external synchronization.

## Formal matrix

Three fresh process runs, alternating policy order, two warm-ups and five measured iterations:

| Model | Case | Arena / baseline | Allocations baseline→Arena | Arena bytes |
|---|---|---:|---:|---:|
| Qwen | prefill T32 B1 | 1.000× | 3135→2415 | 1.16 MB |
| Qwen | prefill T512 B1 | 1.022× | 3495→2895 | 18.61 MB |
| Qwen | prefill T32 B4 | 1.001× | 3020→2420 | 4.65 MB |
| Qwen | decode B1 | 1.031× | 10630→7750 | 0.11 MB |
| Qwen | decode B4 | 0.998× | 10635→7755 | 0.44 MB |
| DeepSeek | prefill T32 B1 | 1.001× | 3515→2815 | 2.11 MB |
| DeepSeek | prefill T512 B1 | 1.020× | 4075→3375 | 33.82 MB |
| DeepSeek | prefill T32 B4 | 1.001× | 3520→2820 | 8.45 MB |
| DeepSeek | decode B1 | 0.998× | 23650→16930 | 0.86 MB |
| DeepSeek | decode B4 | 1.000× | 23515→16935 | 3.43 MB |

All 60 complete-logit files are bit-exact and every decode repeats the fixed expected
tokens. Three rows pass 1.01; none falls below 0.98.

![Complete-model BF16 FFN Arena](../assets/bf16-ffn-arena-model.svg)

## Profiler

Qwen T512 executes exactly 5,642 Kernels in both modes. Kernel duration is 49.07/49.44 ms,
and both direct launch counters are unchanged. Whole-process malloc/free falls from
1,879/1,567 to 1,637/1,327. The small throughput gain is host allocation removal; device
math did not improve.

## Decision

Keep the explicit API, statistics, CLI and runner because they provide the required
complete-model gate and eliminate thousands of logical allocations. Do not enable Arena
globally: seven rows miss 1.01. The evidence supports one new hypothesis only—route
`flattened rows>=512`—because both models improve there. Model-specific R1 routing is not
accepted from one Qwen-only win.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-ffn-arena-model/`](../../../benchmarks/results/2026-08-24-bf16-ffn-arena-model/).
