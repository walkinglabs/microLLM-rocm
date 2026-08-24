# Experiment 176 — route Storage lifetime before routing the model Stream

Status: `keep` as an explicit lifetime primitive; not yet a model execution policy

## Why this exists

Experiment 175 showed that sending model Kernels to a non-default Stream while keeping legacy
temporary-destruction semantics corrupts complete logits. Synchronizing every temporary before
free would restore correctness but serialize the chain. The missing primitive is one region that
retains destroyed allocations until its Stream is complete.

## Contract

`runtime::DeferredHipDeallocationScope`:

- binds one explicit HIP Stream and the current host thread;
- intercepts only same-device HIP deallocations on that thread;
- preallocates a fixed record table, so deallocation remains noexcept;
- is non-copyable, non-movable and non-nestable;
- exposes pending/total blocks and bytes plus overflow flush count;
- `finish()` synchronizes the Stream once, then releases every queued block;
- fixed-capacity overflow performs a safe intermediate Stream synchronization and continues;
- CPU Stream and zero capacity are explicit errors.

It does not select an operator Stream. Callers must still pass `OpContext.stream` themselves.
This separation prevents the unsafe ambient API from Experiment 175 from returning unnoticed.

Logical `engine_peak_bytes` drops when Tensor ownership ends even though the raw allocation is
still physically retained. Therefore `pending_bytes` is a separate mandatory memory metric; the
largest formal row retains 2,080,768 bytes.

## Correctness tests

An eight-node explicit-Stream add chain queues exactly seven 4-float blocks, synchronizes once,
frees them and returns `[8,8,8,8]` with zero payload transfers. A capacity-two test forces two
overflow flushes and still returns `[7,7]`. Nested scope, CPU and zero-capacity failures are
checked. The installed CMake consumer links the public rejection path.

## MI300X matrix

The safe control synchronizes before each old temporary is released. The candidate queues all
old temporaries and synchronizes once. Each cell is a three-process alternating-order median.

| Nodes | 1 element | 4,096 elements | Deferred bytes at 4,096 |
|---:|---:|---:|---:|
| 8 | 2.283× | 2.332× | 114,688 |
| 32 | 2.692× | 2.664× | 507,904 |
| 128 | 2.431× | 2.739× | 2,080,768 |

All 36 processes have exact output, exact `(nodes-1)×elements×4` deferred bytes, no overflow and
zero timed H2D/D2H/D2D. Every row clears the predeclared 1.20 wall gate.

## Profiler attribution

For 32×4096 over ten repetitions:

| Counter | Immediate safe sync | Deferred |
|---|---:|---:|
| Kernel calls | 323 | 323 |
| Kernel launches | 320 | 320 |
| `hipMalloc` / `hipFree` | 322 / 322 | 322 / 322 |
| Stream synchronize calls | 320 | 10 |
| Synchronize duration | 3.260 ms | 0.131 ms |
| Total HIP API calls | 2,901 | 2,291 |

The candidate does not hide work or reuse blocks. It removes 310 synchronization points by
making raw lifetime explicit.

![Deferred HIP deallocation](../assets/deferred-hip-deallocation.svg)

## Decision and next step

Keep the explicit primitive, overflow fallback, counters, benchmark and tests. Do not turn it
into a global allocator mode and do not report the micro speedup as model acceleration.

The next node may combine this lifetime scope with a model Stream scope and rerun the exact tiny
Transformer failure from Experiment 175. Even if correctness returns, model performance and
temporary-byte growth require separate gates; Graph capture still cannot perform synchronous
allocations and remains a later boundary.

Raw evidence is in
[`benchmarks/results/2026-08-24-deferred-hip-deallocation/`](../../../benchmarks/results/2026-08-24-deferred-hip-deallocation/).
