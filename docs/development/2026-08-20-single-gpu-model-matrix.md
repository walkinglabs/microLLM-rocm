# 2026-08-20 — single-GPU model memory/performance matrix

## Observed gap

The end-to-end benchmark already emitted throughput and peak engine bytes, but its
model selector only accepted tiny and Model-S. The HIP CTest gate measured only tiny
generation. There was no one-command table covering multiple model sizes, and the JSON
omitted exact parameter count, weight bytes, current memory, cumulative allocation,
construction time and warm-up time.

## Contract

The first matrix covers repository-owned FP32 profiles so it can run without external
weights:

```text
tiny     5,712 parameters
Model-S  15,586,176 parameters
Model-M  31,334,912 parameters
```

Each profile runs training and cache-backed generation. Performance is recorded, but
only schema, finiteness, model identity, exact parameter/weight size and memory
invariants are pass/fail gates.

## MI300X result

Environment reported by the executable:

```text
device        AMD Instinct MI300X VF
architecture  gfx942:sramecc+:xnack-
HIP runtime   71399004
HIP driver    71399004
dtype         FP32
```

| Model | Mode | Context | Tokens | tokens/s | ms/token | Peak engine MiB |
|---|---|---:|---:|---:|---:|---:|
| tiny | train | 8 | 24 | 4379.90 | 0.228 | 0.142 |
| tiny | generate | 8 | 24 | 860.09 | 1.163 | 0.027 |
| Model-S | train | 2 | 2 | 1.111 | 899.978 | 238.687 |
| Model-S | generate | 4 | 2 | 1.217 | 821.929 | 59.608 |
| Model-M | train | 1 | 1 | 0.528 | 1893.835 | 478.765 |
| Model-M | generate | 4 | 2 | 1.226 | 815.732 | 119.754 |

The dedicated CTest passed all six measurements in 11.30 seconds. CPU schema and
orchestration also passed as part of the 144-test CPU gate.

## Interpretation boundary

These are random-weight, very-short-context smoke measurements. They prove that
performance and memory are captured for each model size; they do not represent model
quality, long-context throughput, steady-state serving, or PyTorch speedup.

The low Model-S/Model-M decode rate exposes a real optimization target. Current cached
decode still has dynamic cache/GQA work that must become preallocated and device-native.
The matrix makes that failure visible rather than treating a successful output as a
performance success.

Engine peak bytes exclude private driver/vendor/runtime allocations. External device
memory tracing remains required for a board-level peak claim.
