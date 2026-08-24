# Experiment 179 — two reusable addresses still do not make an eager model allocator

Status: `keep` explicit beta primitive; reject eager and Graph allocation policies

## Hypothesis

Experiment 177 was correct but slow because non-default Stream work disabled the legacy-default-
Stream-only exact-size cache. HIP exposes stream-ordered allocation: a free can be queued after the
last use, and a later allocation on the same Stream may safely reuse it without a host wait.

AMD documents `hipMallocAsync`/`hipFreeAsync` as Stream-ordered, Linux-implemented Beta APIs. It
also states that capture creates Graph allocation/free nodes. See the
[official allocator guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html).

## Explicit implementation

`runtime::StreamOrderedHipBuffer` is move-only, binds one Stream, enqueues allocation/release and
requires that Stream to outlive the buffer. Capability, default-pool reserved/used high/current,
release threshold and trim APIs are public diagnostics. Ordinary `Storage` is untouched.

Tests cover CPU rejection, capability, move/release, pool high-water, zero size and allocation+
free nodes inside `HipGraphExecutable`. Current ROCm 7.13 reports support. It also returns
`reserved_current=0` after synchronization despite an 8 MiB threshold in the small test, so only
same-Stream pre-sync reuse is claimed.

## Formal matrix

72 fresh processes, 20 timed chains per process:

| Nodes | Async ratio, E1 / E4096 | Graph ratio, E1 / E4096 |
|---:|---:|---:|
| 8 | 0.702× / 0.668× | 0.0476× / 0.0468× |
| 32 | 0.709× / 0.648× | 0.0414× / 0.0395× |
| 128 | 0.649× / 0.619× | 0.0367× / 0.0364× |
| 512 | 0.637× / 0.700× | 0.0396× / 0.0409× |

All outputs are exact. Async uses at most two addresses for every row, but the pool reserves
128 MiB high-water even when two 4-byte buffers are sufficient. Graph uses exactly N addresses
and `3N+1` nodes: N alloc, N free and N+1 add/copy Kernels.

![Stream ordered allocator result](../assets/stream-ordered-allocator.svg)

## Profiler attribution

At 128×4096 over 3 warm-ups + 20 measurements:

| Counter | Deferred | Async eager | Graph |
|---|---:|---:|---:|
| Executed Kernels | 2,971 | 2,971 | 2,971 |
| Kernel duration | 5.60 ms | 7.89 ms | 10.96 ms |
| sync malloc/free | 2,948 / 2,948 | 4 / 4 | 4 / 4 |
| async malloc/free | 0 / 0 | 2,944 / 2,944 | 128 / 128 capture calls |
| host Kernel launches | 2,967 | 2,967 | 129 |
| Graph launches | 0 | 0 | 23 |

Graph removes host submissions but allocation/free nodes serialize costly device-side memory
work. Eager async merely exchanges one API family for another and lengthens the dependency chain.

## Decision

Keep the explicit buffer/capability/Graph conformance and measurement infrastructure. Do not
route Tensor Storage or model temporaries through it. Eager async and captured allocation-node
policies are closed on this runtime.

The remaining distinct direction is a caller-owned activation arena with stable addresses and a
graph-wide liveness plan. It must allocate outside replay and suballocate without per-Tensor HIP
allocation nodes.

Raw evidence is in
[`benchmarks/results/2026-08-24-stream-ordered-allocator/`](../../../benchmarks/results/2026-08-24-stream-ordered-allocator/).
