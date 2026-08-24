# Experiment 173 — HIP Graph wins after the submission-count crossover

Status: `keep` for the caller-owned runtime primitive; `not integrated` for model execution

## Why this is the next boundary

Experiment 172 removed logical Tensor allocations but left all device work and HIP calls in
place. The post-layout Qwen profile still has thousands of Kernel submissions per process.
Unlike another local owner predicate, HIP Graph can change that host-to-device submission
contract while preserving every Kernel.

AMD's [HIP Graph guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
documents capture → instantiate → replay and explicitly forbids synchronous `hipMalloc`,
`hipMemcpy` and `hipFree` during stream capture. AMD's
[graph-safe library table](https://rocm.docs.amd.com/en/develop/reference/graph-safe-support.html)
marks hipBLASLt only partially supported. Therefore the first experiment may not claim that a
whole Transformer is capturable.

## Contract

`runtime::HipGraphExecutable` is move-only and accepts one explicit HIP `Stream` plus a capture
callback. It records node count, instantiates once, checks replay device, and destroys template
state after instantiation. CPU, empty callback, empty graph, undefined use and device mismatch
are explicit errors.

Most importantly, caller-owned addresses must remain valid until the last replay completes.
The wrapper does not guess Tensor lifetime, copy payloads or convert default-Stream model code.

## Failure recovery found on the real runtime

The test intentionally constructs a Tensor inside capture. The synchronous allocation is
rejected as expected, but the first implementation then failed its next legal Kernel with
`operation failed due to a previous error during capture`. Ending capture alone did not clear
the sticky HIP error.

The repaired exception path ends/destroys abandoned capture state, consumes the sticky error,
rethrows the original exception, and then successfully captures and replays two caller-owned
operators on the same Stream. This is the exact fallback behavior the model layer will need.

## MI300X crossover matrix

The workload captures one caller-owned fill plus N in-place-style add-out Kernels. Each row is
the median of three fresh eager/graph processes with alternating order, five warm-ups and 20
measured repetitions.

| Add nodes | Wall speedup, 1 element | Wall speedup, 4,096 elements |
|---:|---:|---:|
| 1 | 0.595× | 0.590× |
| 8 | 0.890× | 0.827× |
| 32 | 1.328× | 1.207× |
| 128 | 1.567× | 1.503× |
| 512 | 1.909× | 1.728× |

All 60 processes have zero error, zero timed H2D/D2H/D2D and exact node count. The declared
`>=1.05×` gate for every N>=32 row passes. One and eight nodes are the required counterexample:
Graph is not a universal “make one Kernel faster” switch.

Median setup/instantiation is about 12–14 ms for most rows. That cost is outside replay timing
and needs many iterations to amortize.

## Profiler attribution

The 128-node, one-element control uses zero warm-up and 20 repetitions:

| Counter | Eager | Graph |
|---|---:|---:|
| Executed Kernel calls | 2,583 | 2,583 |
| `hipLaunchKernel` host calls | 2,580 | 129 capture-only |
| `hipGraphLaunch` calls | 0 | 20 |
| Total traced HIP API calls | 12,990 | 802 |

The graph does not fuse arithmetic or reduce device launches. It compresses repeated host
submission. Profiler absolute durations are heavily instrumented; the uninstrumented
three-process matrix remains the speed gate.

![HIP Graph submission crossover](../assets/hip-graph-submission-crossover.svg)

## Decision and next blocker

Keep the runtime primitive, test, benchmark and matrix runner. Do not enable a model flag yet.
Current model/autograd calls do not propagate `OpContext::stream`, and forward/backward create
temporary Tensor Storage whose owners disappear after capture. Direct capture would either use
the wrong Stream, hit prohibited allocation, or replay dangling pointers.

The next valid node must solve one boundary at a time: first a fixed caller-owned operator region
that includes a real hipBLASLt shape, then explicit execution-context propagation, then stable
activation/gradient lifetime. “Wrap `model.loss()` in begin/end capture” is now a disproven plan.

Raw evidence is in
[`benchmarks/results/2026-08-24-hip-graph-runtime/`](../../../benchmarks/results/2026-08-24-hip-graph-runtime/).
