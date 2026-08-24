# Experiment 184 — rows≥512 selective BF16 FFN Arena

Status: `keep`

## Hypothesis

Universal Arena reduced allocations but passed only three of ten speed rows. Both models improved
at flattened rows 512. Can one model-independent row threshold keep those wins while making short
prefill/decode exactly the old path?

## Contract

`set_bf16_ffn_arena_enabled(true, minimum_rows)` accepts only positive thresholds. Each FFN call
increments either eligible or bypassed; a bypass cannot create an entry or backing allocation.
The CLI exposes the threshold and all counters. Model names never participate in dispatch.

## Formal matrix

The exact same 60 fresh processes, revisions, cases, warm-ups and repetitions as Experiment 183:

| Model | Case | Selective / baseline | Entry bytes | Allocation calls |
|---|---|---:|---:|---:|
| Qwen | T32 B1 | 1.001× | 0 | 3135→3135 |
| Qwen | T512 B1 | 1.019× | 18.61 MB | 3495→2895 |
| Qwen | T32 B4 | 1.003× | 0 | 3020→3020 |
| Qwen | decode B1 | 1.005× | 0 | 10630→10630 |
| Qwen | decode B4 | 1.001× | 0 | 10635→10635 |
| DeepSeek | T32 B1 | 0.999× | 0 | 3515→3515 |
| DeepSeek | T512 B1 | 1.022× | 33.82 MB | 4075→3375 |
| DeepSeek | T32 B4 | 1.001× | 0 | 3520→3520 |
| DeepSeek | decode B1 | 1.003× | 0 | 23650→23650 |
| DeepSeek | decode B4 | 0.999× | 0 | 23515→23515 |

Every complete logit is bit-exact; decode tokens match. Eligible rows have entry=1 and zero bypass.
All eight bypass rows have entry/capacity/eligible=0 and positive bypass counts.

![Selective BF16 FFN Arena](../assets/bf16-ffn-arena-selective.svg)

## Profiler

Qwen T512 executes 5,642 Kernels in both modes. Direct launch counts remain 4,000 and 1,519.
malloc/free falls 1,879/1,567→1,637/1,327. Kernel duration is 51.06/47.98 ms in this profiling
run; the formal throughput conclusion comes from the three fresh unprofiled processes.

## Decision

Keep the explicit threshold and `rows>=512` policy evidence. It improves both official models and
proves all other declared cases use the original path. It remains opt-in; broader checkpoint and
hardware evidence is required before an unconditional default. The next distinct allocation region
is shared-cast BF16 Q/K/V at long prefill, not another FFN threshold between observed points.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-ffn-arena-selective/`](../../../benchmarks/results/2026-08-24-bf16-ffn-arena-selective/).
