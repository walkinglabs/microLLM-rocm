# Experiment 187 — even the largest allocation source is not the speed bottleneck

Status: `discard` model policy; keep caller-owned primitive

## Hypothesis

Experiment 186 identifies Attention core as 43.6%–53.0% of logical allocated bytes. If host
allocation is still limiting long prefill, removing this exact family should improve both models.

## Exact liveness

```text
scaled Q ─┐
expanded K ─ QK → probabilities(in-place softmax) ─ PV → output
             │
             └ expanded K slot becomes expanded V after QK submission
```

The caller owns scaled Q, one reusable expanded K/V slot, probabilities and output. Compared with
the existing long path, this replaces `probability + 5×hidden` logical allocations per layer with
one persistent `probability + 3×hidden` backing shared across blocks.

## Formal complete-model result

| Model | T512 core/baseline | Backing | Allocation calls | Peak |
|---|---:|---:|---:|---:|
| Qwen | 1.004× | 20.19 MB | 2895→2295 | +2.75 MB |
| DeepSeek | 1.002× | 22.02 MB | 3375→2675 | +4.72 MB |

Both eligible rows fail the 1.01 keep gate. Eight short/decode cases create no core entry and stay
0.993×–1.004×. All 60 complete logits and fixed decode tokens are exact.

![Attention core Arena discard](../assets/attention-core-arena-discard.svg)

## Profiler

Qwen T512 keeps exactly 5,642 Kernels, 4,000 ordinary launches and 1,519 extended launches.
malloc/free falls 1,637/1,327→1,395/1,087; Kernel time is 47.67/46.78 ms. Removing the measured
largest allocation source still buys only 0.4%/0.2% in formal end-to-end runs.

## Decision

Reject model routing and keep it default-off. Retain `causal_gqa_attention_out_`, liveness tests
and the diagnostic seam. The allocation-reuse track is now saturated: QKV and the measured largest
core family both reduce calls without reaching 1.01. The next candidate must improve Attention
device math—exact FP32 QK/PV algorithms or a genuinely fused implementation—not more Storage.

Raw evidence:
[`benchmarks/results/2026-08-24-attention-core-arena-discard/`](../../../benchmarks/results/2026-08-24-attention-core-arena-discard/).
