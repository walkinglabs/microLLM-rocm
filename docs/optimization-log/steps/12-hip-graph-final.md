# Step 12 — HIP Graph, final matrix and publication

Status: `planned`

Experiment 021 showed that removing 30k repeated `hipSetDevice` calls can improve the
instrumented timeline while regressing every uninstrumented workload. Future scheduling
work must preserve external device-state semantics and pass the fixed model matrix; API
count alone is not a keep gate.

## Prerequisites

Do not begin until:

- allocator provides stable addresses;
- KV Cache is preallocated;
- operator plans/workspaces are cached;
- measured step has no dynamic host data decisions that prevent replay.

## HIP Graph experiment

- capture one fixed train step separately from one decode step;
- instantiate once;
- update allowed inputs without rebuilding graph;
- compare launch/API timeline before/after;
- preserve an eager fallback.

## Final matrix

- FP32 fixed matrix and running-best curve;
- optional BF16 and FP8 separate curves;
- context 1/32/128/512;
- batch sensitivity;
- tokens/s, ms/token and memory;
- operator and end-to-end results;
- Qwen/DeepSeek exact correctness;
- one stable failure;
- PyTorch raw data generated in the recorded environment.

## Completion rule

Selected-matrix parity is achieved only when the committed raw data and validator prove
the target. A rounded README table is not evidence.

## Publication

Update the living blog with:

```text
what we believed
what we changed
what actually happened
which experiments failed
which explanation was falsified
where PyTorch remains faster
which claims are model/shape specific
```
