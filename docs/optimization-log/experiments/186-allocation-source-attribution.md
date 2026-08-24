# Experiment 186 — stop guessing allocation sources

Status: `keep` diagnostic; select `attention.core`

## Why

QKV Arena removed hundreds of allocations but improved only 0.4%–0.5%. The next persistent
Tensor family must come from measured source and size attribution rather than code-reading alone.

## Instrumentation contract

`ScopedAllocationSource` uses a fixed enum, restores nested tags and is thread-local. When
diagnostics are disabled, construction performs one branch and does not change the active tag.
Enabled logical allocations aggregate exact `(source, device, bytes)` records; cache reuse remains
an allocation request and is counted. Diagnostics never claim backend malloc count.

CLI diagnostics require one prefill and zero warm-up. This prevents load, preparation and warm-up
from being mixed with the selected forward. Three fresh processes must have identical records.

## Formal result

| Source | Qwen calls / bytes | DeepSeek calls / bytes |
|---|---:|---:|
| attention.core | 144 / 572.52 MB | 168 / 792.72 MB |
| attention.projection | 168 / 135.27 MB | 196 / 278.92 MB |
| attention.layout | 120 / 106.95 MB | 140 / 220.20 MB |
| attention.output | 48 / 66.06 MB | 56 / 132.12 MB |
| four norm/residual families | 96 / 176.16 MB | 112 / 352.32 MB |
| retained FFN Arena | 1 / 18.61 MB | 1 / 33.82 MB |
| model embedding/norm/head | 3 / 4.28 MB | 3 / 6.90 MB |

Total is 580 calls / 1.080 GB for Qwen and 676 / 1.817 GB for DeepSeek. The distribution is
bit-for-bit identical across all three processes per model.

![Allocation source attribution](../assets/allocation-source-attribution.svg)

### Exact Attention core sizes

- Qwen: 14,680,064 bytes ×24 plus 1,835,008 bytes ×120;
- DeepSeek: 12,582,912 bytes ×28 plus 3,145,728 bytes ×140.

The large family equals the causal score/probability scale; the repeated hidden-width family
contains the other Attention core intermediates. Attention core accounts for 53.0%/43.6% of
logical allocated bytes.

## Decision

Keep diagnostics and make `attention.core` the next optimization target. The next node must map
its exact liveness and caller-owned outputs; QKV/FFN/storage guesses are closed until a new profile
changes this distribution.

Raw evidence:
[`benchmarks/results/2026-08-24-allocation-source-attribution/`](../../../benchmarks/results/2026-08-24-allocation-source-attribution/).
