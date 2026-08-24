# Experiment 191 — wider search makes steady grouped QKV pass

Status: `keep` exact warmed policy; default remains off

## Why reopen a rejected model policy

Experiment 190 screened only 16 supported grouped algorithms. Qwen passed the model gate, while
DeepSeek was neutral despite a 1.225× operator gain. The only new variable allowed here is a wider
correctness-first solution search for the same exact shape and environment.

## Expanded operator gate

Every candidate runs complete Q/K/V outputs before timing. Three fresh processes screen 64
supported algorithms; direct grouped FP32 output controls remain unsupported.

| Model | Index | Event | Wall | Per-call reinitialize |
|---|---:|---:|---:|---:|
| Qwen | 64713 | 2.010× | 1.569× | 0.954× |
| DeepSeek | 64755 | 1.692× | 1.493× | 0.958× |

DeepSeek's stronger index is stable across three independent 64-candidate probes. Qwen also moves
to 64713. Projection Max/RMS remains at most `2.44e-4/1.08e-4`.

## Plan initialization failure and repair

The first implementation initialized one GroupedGemm per block. A cold T512 forward took about
5.7 seconds. Caching only the algorithm object did not help: the expensive operation was grouped
initialization itself.

The repaired design initializes one kernel per exact shape/index/device/Stream with device user
arguments enabled. Each block stores only a small device argument record containing its weight
pointers and shared Arena addresses. Diagnostics prove one algorithm and one kernel entry,
23/27 kernel hits, 24/28 pointer plans and only 0.46/0.67 ms total argument setup.

The remaining first kernel setup is still 207.9/203.7 ms in the formal processes, above the
declared 100 ms one-shot/default gate. Under rocprof instrumentation a cold setup can be several
seconds, so it cannot be hidden inside warm-up.

## Final complete-model gate

Three fresh processes per model/policy use final indices, two warm-ups, five measurements and
reversed even-run order. An accidental same-GPU concurrent run was interrupted and excluded.

| Model | Speedup | Max/RMS logits | Peak | Allocations |
|---|---:|---:|---:|---:|
| Qwen | 1.0458× | 0.10881 / 0.02360 | 1.0034× | 2895→2415 |
| DeepSeek | 1.0295× | 0.07200 / 0.01255 | 1.0017× | 3375→2815 |

Both top tokens match and all logits are finite. The steady correctness/performance/memory policy
passes; the setup gate fails.

## Device explanation

Phase-delta traces show real device work reduction. Qwen GEMM calls/time fall
`217→169` and `3.194→2.807 ms`; DeepSeek falls `253→197` and `6.479→6.140 ms`.
Extra BF16→FP32 casts offset part of the gain. Total Kernel time improves 1.019×/1.021×,
consistent with rather than sufficient to explain the uninstrumented throughput result.

![Expanded BF16 grouped QKV](../assets/bf16-grouped-qkv-expanded.svg)

## Decision

Keep the explicit warmed policy and final runner defaults (`64713/64755`) for this exact MI300
environment. Do not enable it for one-shot/default inference because setup exceeds 100 ms, and do
not extrapolate indices to another ROCm/hipBLASLt build. A future serving API may prewarm before
admission and then re-evaluate end-to-end request latency.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-grouped-qkv-expanded/`](../../../benchmarks/results/2026-08-24-bf16-grouped-qkv-expanded/).
