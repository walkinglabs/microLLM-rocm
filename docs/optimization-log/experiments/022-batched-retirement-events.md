# Experiment 022 — batched allocator retirement Events

Status: `keep`

## Observation

The exact-size pool recorded one completion Event for every destroyed Tensor. A short
DeepSeek trace contained 8,993 Event create/record calls, and rocprof amplified their
API cost.

## Design

- default-Stream frees enter a per-device pending list;
- every eight blocks share one timing-disabled Event recorded after all preceding work;
- each retired block holds shared ownership of that Event;
- a block is reusable only after the shared Event reports ready;
- `synchronize(device)` flushes an incomplete batch before synchronizing;
- constructing a non-default Stream flushes pending work and permanently disables pool
  reuse, preserving the existing conservative external-Stream rule;
- Event creation/record failure frees the pending blocks and fixes allocator counters.

The batching changes only lifetime proof, not Tensor values or operation order.

## Correctness

- CPU debug: `157/157` pass;
- ASan/UBSan: `155/155` pass;
- MI300X/gfx942 HIP: `57/57` pass;
- new test allocates eight equal blocks, retires them behind one completion boundary and
  proves all eight are reused without another backend allocation;
- allocator stress, exception destruction, asynchronous copies, explicit Streams,
  TensorView, full CPU/HIP graph and official model tests pass;
- official tokens, loss and observed AdamW parameters remain exact.

## Profiler result

```text
Event create calls       8,993 → 1,124
Event record calls       8,993 → 1,124
Event record API time    24.39 → 1.95 ms
Event destroy calls      8,577 → 1,071
Kernel launches          5,908 → 5,908
instrumented DeepSeek    29.27 → 51.22 token/s
```

The Event counts fall by almost exactly 8×. Kernel launch count is unchanged, which
supports the allocator explanation.

## Three-process medians

| Workload | Experiment 018 | Candidate median | Change | PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 112.43 | 154.78 token/s | +37.7% | 3.0158× |
| Qwen generate | 154.60 | 201.39 token/s | +30.3% | 2.8695× |
| DeepSeek train | 67.41 | 81.99 token/s | +21.6% | 3.1262× |
| DeepSeek generate | 58.32 | 75.24 token/s | +29.0% | 1.2058× |

The displayed training candidate medians are compared with Experiment 016 training
medians because Experiments 017–018 did not modify or rerun training. All four candidate
samples come from the same three full matrices.

```text
score       1.845199 → 2.389841
```

Backend allocation counts remain within the existing process variation and engine peak
bytes do not increase.

## Decision

`keep`. Correctness and failure paths pass, the trace shows the intended 8× API
reduction, all four repeated workload medians improve, DeepSeek generation crosses the
fixed PyTorch reference, and memory does not regress.

Raw evidence is in [022-data](022-data/README.md).
