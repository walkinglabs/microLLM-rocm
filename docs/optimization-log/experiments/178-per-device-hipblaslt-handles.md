# Experiment 178 — hipBLASLt handles belong to a device

Status: `keep`

## Stable failure

After Experiment 177, the RCCL suite exposed five failures. Collectives passed, but every test
that ran a Transformer on GPU 0 and then GPU 1 failed inside a vendor GEMM with
`invalid device ordinal`. A detached fresh build of pre-node `adcd642` reproduced the same error,
so this was an existing multi-GPU backend-ownership bug, not a Stream-scope regression.

## Hypothesis

`optimized.cpp` used three process/thread static hipBLASLt handles: general GEMM, BF16/FP8 GEMM
and Attention layouts. The first call created a handle under the current GPU. Later rank-local
calls changed device but reused that handle.

The minimal fix is not a global mutex or a device-wide synchronization. Handle ownership must be
explicitly keyed by device, like the existing BF16/Attention plan cache keys.

## Change and tests

`handle_for_device(Device)` selects the device first, then returns a thread-local handle stored by
device index. General FP32/FP16/BF16 matmul, BF16/FP8 specialized matmul and all Attention layout
calls use it.

A new HIP test alternates `0 → 1 → 0 → 1`, runs both FP32 and BF16 GEMM, checks every output, and
proves the BF16 cache has two entries/two misses/two hits. This smaller test fails at the true
ownership boundary without requiring RCCL.

## Results

| Gate | Before | After |
|---|---:|---:|
| RCCL multi-GPU tests | 6/11 | 11/11 |
| RCCL Config consumers | 2/2 | 2/2 |
| Alternating device FP32/BF16 | fail indirectly | pass |

The single-GPU T512 regression matrix uses 12 new processes and previous-revision Experiment 177
raw rows:

| Workload | Throughput ratio | Output contract |
|---|---:|---:|
| Qwen inference | 1.023× | exact |
| Qwen training | 1.000× | exact |
| DeepSeek inference | 0.998× | exact |
| DeepSeek training | 1.001× | exact |

![Per-device hipBLASLt handle result](../assets/per-device-hipblaslt-handles.svg)

## Decision

Keep per-device handles. The fix restores two-rank training without changing algorithms, Kernel
math or public model policy, and its worst single-GPU ratio is 0.9979.

This closes handle ownership. It does not claim production one-process-per-GPU launch semantics,
gradient-ready overlap or cross-node support; those remain separate multi-GPU tasks.

Raw evidence is in
[`benchmarks/results/2026-08-24-per-device-hipblaslt-handles/`](../../../benchmarks/results/2026-08-24-per-device-hipblaslt-handles/).
