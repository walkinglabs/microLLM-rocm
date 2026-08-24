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

## Experiment 174 vendor-GEMM result

`matmul_out_` now gives hipBLASLt a stable caller-owned output, and the current MI300X runtime
captures each warmed GEMM as exactly one node with bit-exact replay. Compatibility is no longer a
guess.

Performance is still rejected. Qwen at 1/8/32 repeated T512 GEMMs reaches
0.906×/0.995×/1.022×; DeepSeek reaches 0.902×/0.989×/0.990×. Profiler reduces host module-launch
calls while leaving all GEMM Kernels intact. The next region must be heterogeneous—small Kernels
around real GEMMs—not another repetition-count sweep of one vendor operation.

## Experiment 175 Stream-propagation failure

A nested thread-local Stream scope routed the existing default-`OpContext` model call tree and
passed caller-owned tests. The tiny Transformer immediately failed all 64 logits in three runs:
Max/RMS was 1.412/0.475, 3.846/0.931 and 1.412/0.475.

The failure locates the next prerequisite. Temporary Storage ownership ends before queued
non-default-Stream consumers complete. The scoped API is removed; synchronizing every destructor
is rejected. Deferred release or a planned activation arena must precede any model-wide Stream
or Graph retry.

## Experiment 176 lifetime primitive

A fixed-capacity, non-nestable deferred-release scope now queues destroyed raw HIP allocations
until one explicit Stream synchronization. The 8/32/128-node × 1/4096-element matrix is exact and
improves every safe-control row by 2.28×–2.74×. Profiler reduces synchronization 320→10 while
leaving allocations, frees and Kernels unchanged.

The cost is explicit: the largest row retains 127 blocks / 2,080,768 bytes. This primitive does
not route model operators. Experiment 175 may only be retried by combining the two contracts and
reporting complete logits, time and pending bytes.

## Experiment 177 safe model retry

`ScopedDeferredHipStream` now binds default `OpContext`, runtime strided-copy and temporary raw
lifetime to one Stream. The original three-run tiny failure becomes exact; complete forward/
backward gradients and 24 official model pairs also pass exactly.

The execution policy is rejected. Across Qwen/DeepSeek, inference/training and T32/T512, candidate
throughput is 0.125×–0.862× of legacy and deferred physical memory reaches 15.6 GB. Qwen T512
profiler work is unchanged at 2,751 Kernels, but whole-process malloc/free calls rise from
1,180/867 to 2,559/2,557. Model capture is now blocked on ordered allocation or an activation
arena, not on Stream correctness.

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
