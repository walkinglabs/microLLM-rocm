# Experiment 190 — grouped QKV wins locally, not jointly

Status: `keep` explicit pointer-stable primitive; `discard` model default

## Measurement correction first

A whole-process T512 trace says BF16 weight cast-transpose dominates, but that Kernel runs during
one-time preparation. We profiled load+one and load+six prefills, subtracted the first, and divided
by five. Incremental Kernel time is 5.960 ms Qwen and 10.473 ms DeepSeek. hipBLASLt GEMMs account
for 53.6% and 61.9%; DeepSeek's 84 per-forward Q/K/V projection calls alone take 2.661 ms.

The old long-sequence fused Attention Kernel was already rejected by Experiment 56, so repeating
it was not a new hypothesis. The trace instead reopened Experiment 13's explicitly deferred case:
BF16 GroupedGemm at prefill M=512.

## Operator contract

Three Q/K/V problems share one BF16 input but retain separate weights and outputs. Direct grouped
FP32 output returns no supported solution, including equal-width controls. Grouped BF16 output is
supported. The candidate therefore runs one grouped operation into three caller-owned BF16 buffers,
then casts each result into its existing FP32 output.

Sixteen supported candidates run complete-output finite/Max/RMS checks before timing. Three fresh
processes per model are intersected by behavior. A pointer-stable initialized plan wins:

| Model | Event | Wall | Reinitialize each call | Solution |
|---|---:|---:|---:|---:|
| Qwen | 1.881× | 1.553× | 0.908× | 64699 or 64700 |
| DeepSeek | 1.225× | 1.174× | 0.815× | 64701 |

The counterexample is part of the design: descriptor setup is larger than the saved submission
time. The engine caches one initialized plan per exact shape, environment, device, Stream and all
input/weight/output pointers. QKV Arena makes activation/output pointers stable; block-specific
weight pointers intentionally create 24/28 plans.

## Complete-model gate

The formal baseline retains selective BF16 FFN Arena. Candidate additionally enables long-row QKV
Arena and exact grouped indices. Three fresh processes per model/policy use two warm-ups, five
measurements and reversed even-run order.

| Model | Plans hit/miss | Dispatches | Speedup | Max/RMS logits | Peak |
|---|---:|---:|---:|---:|---:|
| Qwen | 144/24 | 168 | 1.0317× | 0.09360 / 0.01978 | 1.0034× |
| DeepSeek | 168/28 | 196 | 1.0015× | 0.06300 / 0.02044 | 1.0017× |

Both remain finite, preserve the top token, stay inside Max 0.25/RMS 0.05 and reduce logical
allocations by 480/560. Qwen passes performance; DeepSeek does not reach 1.01.

![BF16 grouped QKV](../assets/bf16-grouped-qkv.svg)

## Decision

Keep the explicit exact-environment registry, pointer-stable plan cache, operator probe, counters
and CLI experiment. Do not enable grouped QKV by default and do not add a model-name branch.
This hypothesis explains a Qwen gain but fails the required two-model rule. A future retry needs a
model-independent shape predicate supported by more checkpoints, or a backend path that improves
the width-1536 family enough to move DeepSeek end to end.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-grouped-qkv/`](../../../benchmarks/results/2026-08-24-bf16-grouped-qkv/).
