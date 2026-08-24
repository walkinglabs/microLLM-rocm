# Experiment 180 — stable arena addresses make Graph useful again

Status: `keep` explicit arena and arena-Graph foundation; model integration remains gated

## Hypothesis

Experiment 179 showed why allocation APIs cannot live inside eager model calls or Graph replay.
The remaining design allocates once outside replay, plans overlapping lifetimes on the host, and
captures only computation over stable offsets.

## Contract

`runtime::HipActivationArena` owns one HIP backing allocation and one explicit Stream. It is
non-copyable/non-movable so base addresses never change. `allocate_slice` enforces nonzero size,
power-of-two alignment and bounds. `reset_plan()` only resets the host planning cursor; callers
must prove lifetimes before reusing a slice. Destruction synchronizes the bound Stream before
freeing the backing allocation.

Tests cover CPU/zero/alignment/overflow, stable reset addresses, two aligned slices and an 8-node
Graph. The captured Graph has exactly nine Kernel nodes—not allocation/free nodes—and replays
exactly.

## Formal MI300X matrix

72 fresh processes, 20 timed chains each:

| Nodes | Arena eager, E1 / E4096 | Arena Graph, E1 / E4096 | Graph break-even replays |
|---:|---:|---:|---:|
| 8 | 1.071× / 1.072× | 1.315× / 1.314× | 1,280 / 1,174 |
| 32 | 1.369× / 1.359× | 2.187× / 2.047× | 171 / 173 |
| 128 | 1.480× / 1.455× | 2.668× / 2.561× | 40 / 40 |
| 512 | 1.595× / 1.768× | 2.953× / 3.066× | 10 / 9 |

All eight eager and Graph rows pass the 1.05 hot-path gate. Setup remains 14–16 ms, so short
graphs need more than a thousand replays; reporting only replay latency would be misleading.

![Activation arena result](../assets/activation-arena.svg)

## Profiler

At N128×4096:

| Counter | Deferred | Arena eager | Arena Graph |
|---|---:|---:|---:|
| Executed Kernels | 2,971 | 2,971 | 2,971 |
| Kernel duration | 5.60 ms | 5.81 ms | 12.04 ms |
| synchronous malloc/free | 2,948 / 2,948 | 5 / 5 | 5 / 5 |
| host Kernel launches | 2,967 | 2,967 | 129 |
| Graph launches | 0 | 0 | 23 |

Instrumented Graph Kernel time rises, but uninstrumented wall improves because host submission
and allocator calls disappear. Both views are retained.

## Decision

Keep the arena, liveness contract, tests, benchmark and arena-Graph candidate. Do not yet expose a
model toggle: the model needs a shape-specific activation plan, caller-owned outputs and a fixed
heterogeneous region. The next node must map one real Transformer block or inference region into
arena offsets and include setup amortization.

Raw evidence is in
[`benchmarks/results/2026-08-24-activation-arena/`](../../../benchmarks/results/2026-08-24-activation-arena/).
