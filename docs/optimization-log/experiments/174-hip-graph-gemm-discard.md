# Experiment 174 — real hipBLASLt GEMMs do not inherit the tiny-Kernel Graph win

Status: `discard` for repeated-GEMM Graph routing; `keep` for caller-owned output infrastructure

## Hypothesis

Experiment 173 shows a `1.21×–1.91×` replay win once a chain contains at least 32 small
repository-owned Kernels. The next required boundary is a real vendor call. If hipBLASLt is
capture-safe and repeated Qwen/DeepSeek T512 GEMMs reduce host submission enough, Graph should
improve both official shapes at eight and 32 calls.

The keep gate was fixed before running: complete output bit-exact, stable caller address, zero
timed payload transfers, one captured node per GEMM, and wall speedup >=1.02 for every shape at
8/32 calls.

## Caller-owned matmul contract

`matmul_out_` validates output shape/dtype/device/contiguity and rejects any input Storage alias.
The explicit hipBLASLt path writes beta-zero results directly into caller Storage. Readable CPU
keeps the ordinary reference; readable rank-two HIP can also write directly.

CPU checks hand values, transpose, shape/dtype and alias failures. HIP checks complete output,
address preservation and zero H2D/D2H/D2D. A separate Graph test warms hipBLASLt, captures the
caller-owned call, replays twice and compares all 1,024 values. The mathematical function remains
the existing `matmul`, whose PyTorch forward/shape matrix is unchanged; the new test surface is
ownership and capture behavior.

## Formal MI300X matrix

Each row is the median of three fresh eager/graph processes, alternating order, three warm-ups and
10 measured repetitions.

| Shape | 1 call | 8 calls | 32 calls |
|---|---:|---:|---:|
| Qwen 512×896×896 | 0.906× | 0.995× | 1.022× |
| DeepSeek 512×1536×1536 | 0.902× | 0.989× | 0.990× |

All 36 processes are bit-exact, address-stable and transfer-free. Captured nodes equal requested
calls. Only Qwen at 32 barely clears 1.02; DeepSeek rejects the cross-shape policy.

## Profiler rebuttal

DeepSeek 32 calls × 10 repetitions:

| Counter | Eager | Graph |
|---|---:|---:|
| Executed Kernels | 322 | 322 |
| Kernel duration | 8.401 ms | 8.601 ms |
| `hipExtModuleLaunchKernel` | 321 | 33 capture/setup |
| `hipGraphLaunch` | 0 | 10 |
| `hipSetDevice` | 661 | 97 |
| All traced HIP API calls | 48,217 | 45,077 |

Host submission is compressed, so capture works. But the wide GEMM already spends most useful
time in device computation; Graph adds replay scheduling without removing that work. The tiny
Kernel crossover from Experiment 173 was real and still does not generalize to this vendor shape.

![HIP Graph GEMM counterexample](../assets/hip-graph-gemm-discard.svg)

## Decision

Reject repeated identical hipBLASLt Graph routing as a model strategy. Keep `matmul_out_`, its
alias/shape tests and the capture conformance test because stable vendor outputs are a prerequisite
for any larger caller-owned region.

The next Graph candidate must capture a heterogeneous model region where many small elementwise/
reduction Kernels surround GEMMs. It also requires explicit Stream propagation and planned
activation lifetime. Repeating the same independent GEMM more times is now a closed search.

Raw evidence is in
[`benchmarks/results/2026-08-24-hip-graph-gemm/`](../../../benchmarks/results/2026-08-24-hip-graph-gemm/).
