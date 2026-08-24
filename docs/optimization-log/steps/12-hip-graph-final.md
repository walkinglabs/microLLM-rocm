# Step 12 — HIP Graph, final matrix and publication

Status: `in progress` — caller-owned runtime primitive retained in Experiment 173

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

## Experiment 173 prerequisite result

The first prerequisite is now executable rather than aspirational. A move-only runtime wrapper
captures explicit-Stream work over caller-owned addresses, restores the Stream after a
capture-unsafe synchronous allocation, and exposes exact node count.

The 1/8/32/128/512-node × 1/4096-element MI300X matrix proves a real crossover: Graph loses below
32 nodes and improves all 32+ rows by 1.207×–1.909×. rocprofv3 leaves Kernel work unchanged while
compressing 2,580 eager launch calls into 129 capture calls plus 20 graph replays for the
128-node control.

Model capture remains unfinished for two measured reasons: model/autograd does not propagate one
explicit Stream, and temporary Tensor allocation is prohibited/unsafe during capture. The next
step is a real vendor-GEMM caller-owned region, followed by execution-context propagation and
stable activation lifetime. This step is not complete until a fixed model step passes.

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
