# Experiment 157 — AdamW correctness before timing

## Question

Experiment 049 retained Scalar and Vectorized AdamW as explicit implementations because
forcing Vectorized regressed every official Qwen training row. It still left three unsafe
gaps: `Auto` had no exact selection key, the benchmark checked only two parameter samples,
and an experimental timing could not be persisted without hand-written policy code.

Can AdamW use the same evidence protocol as matmul without silently enabling a slower
global implementation?

## Contract

The new tuner receives immutable initial parameter, gradient, first moment, second moment,
and optional BF16 mirror tensors. For every candidate it:

1. clones state while preserving the 16-byte alignment class;
2. executes exactly one update against the Scalar reference;
3. compares **every element** of parameter, both moments, and mirror in bounded chunks;
4. rejects non-finite or over-tolerance state before creating timing evidence;
5. records default-Stream HIP Event and wall P50/P95 only after the state gate passes;
6. leaves the caller's tensors and live registry unchanged.

The exact key contains element count, all four FP32 dtypes, mirror presence, each state
pointer's 16-byte alignment, GPU architecture, HIP runtime/driver versions, and operation
mode. Persistent JSONL replacement is transactional; stale environment rows are ignored.
Only `register_adamw_autotune_winner()` accepts a screened result.

## Executed matrix

Five cases ran in three fresh processes with 3 warm-ups and 20 measured updates. All
supported candidates passed complete state with zero observed Max/RMS difference.

| Case | Mirror | Vectorized / Scalar Event P50 | Process recommendations | Decision |
|---|---:|---:|---|---|
| tail 4,099 | yes | 1.000× | vector/scalar/scalar | noise-level tie |
| tail 4,099 unaligned | no | unsupported | scalar/scalar/scalar | rejected before timing |
| 802,816 | yes | 0.860× | scalar/scalar/scalar | regression |
| Qwen embedding 136,134,656 | no | 0.959× | scalar/scalar/scalar | regression |
| DeepSeek embedding 233,373,696 | no | 1.010× | scalar/vector/vector | below 1.05 gate |

The unaligned Vectorized candidate reports `supported=false`, complete-state failure, and
exactly zero Event/wall timing. An explicit Scalar acceptance writes one environment-keyed
cache entry and reloads on the same MI300 environment.

![AdamW correctness before timing](../assets/adamw-correctness-before-timing.svg)

## End-to-end guard

The empty-registry Auto fast path adds one atomic read per update. Three tiny FP32 T128/B8
training processes have median 231,046.812 token/s versus the previous same-machine
231,939.503 token/s: `-0.38%`, treated as neutral. Loss and engine peak remain at the
retained values.

## Decision

Keep the correctness-first registry, cache, CLI, full-state tests, and raw runner. Keep
the default `Auto` fallback on Scalar. No aligned case clears the 1.05 operator gate, so no
Vectorized winner is committed as a model policy and no end-to-end speedup is claimed.

This fresh-process matrix supersedes Experiment 049's single-process operator samples for
dispatch decisions, but does not erase that historical evidence. The disagreement itself
is a warning that one micro timing is not a durable global policy.

Raw evidence is in
[`benchmarks/results/2026-08-23-adamw-correctness-before-timing/`](../../../benchmarks/results/2026-08-23-adamw-correctness-before-timing/).
