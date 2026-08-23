# Experiment 166 — interleaved plan cache wins the operator and loses the model gate

## Hypothesis

The new P×V/dP/dV path created three matrix layouts and one matmul description on every
call. A thread-local cache keyed by `{mode,H,T,D,device}` can reuse immutable objects without
caching pointers, algorithms or workspace.

The production default must remain unchanged unless both official T512 models improve at
least 1.01×. Complete output, loss, parameter and peak gates remain mandatory.

## Correctness and routing

- CPU/no-library reports an empty disabled cache;
- HIP first P×V/dP/dV calls create three exact entries and three misses;
- repeating all three gives three hits and bit-exact output;
- a different shape creates a fourth entry;
- explicit false creates/destroys ephemeral plans while stats remain zero;
- B2 HIP operations make no payload transfer;
- the package consumer links the public stats/clear symbols.

The one-step official route proof reports:

```text
Qwen cached:     entries=3 misses=3 hits=69
DeepSeek cached: entries=3 misses=3 hits=81
uncached:        entries=0 misses=0 hits=0
```

These counts exactly equal three modes and `layers×3 - 3` remaining calls.

## Operator matrix

Four shapes × two policies × three fresh processes produce 24 complete-output rows.
Every cached process reports `1 miss + 23 hits`; every uncached process reports empty
stats. Max/RMS error and timed H2D/D2H are zero.

| Shape | Event speedup | Wall speedup |
|---|---:|---:|
| B2 H2 T3 D2 | 1.098× | 1.053× |
| B1 H14 T128 D64 | 1.181× | 1.185× |
| Qwen T512 | 1.153× | 1.067× |
| DeepSeek T512 | 1.074× | 1.069× |

## Same-binary model rebuttal

Both layout fusions stay enabled. Only `--attention-layout-plan-cache true/false` changes.
Each model/policy uses three fresh B1 T512 processes, one warm-up and two measured steps.

| Model | Uncached | Cached | Speedup | Peak ratio | Loss relative diff |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 15,179.27 | 15,029.99 tok/s | 0.9902× | 1.000 | 0.0154% |
| DeepSeek Distill 1.5B | 6,327.45 | 6,330.86 tok/s | 1.0005× | 1.000 | 0 |

Allocation counts and the observed parameter are equal. Both strided-copy counters remain
zero. The operator result therefore does not survive the full training schedule.

The route-smoke's 1.02× first-step result is not a counterargument: it includes lazy setup,
uses one process and is explicitly marked `smoke`. The warmed three-process comparison is
the decision evidence.

![Attention layout plan cache discarded](../assets/attention-layout-plan-cache-discard.svg)

## Decision

Reject as a production optimization and set both engine and CLI defaults to false. Retain
the exact cache/statistics and explicit control as diagnostic infrastructure, just as the
standard GEMM solution tuner remains available after a model-policy rejection. A future
candidate needs a different execution boundary, not another descriptor cache.

Raw evidence is in
[`benchmarks/results/2026-08-23-attention-layout-plan-cache/`](../../../benchmarks/results/2026-08-23-attention-layout-plan-cache/).
