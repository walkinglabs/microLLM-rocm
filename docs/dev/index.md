# Developer guide

microLLM-rocm is organized as a small runtime with explicit component boundaries. Each
component has a public interface under `include/microllm`, an implementation under
`src`, and corresponding tests under `tests`.

## Start here

1. [Build the repository](build.md).
2. [Run the appropriate test suite](testing.md).
3. Read the [repository layout and dependency rules](repository-layout.md).
4. For operator work, follow [operator development](operator-development.md).
5. For performance work, follow [profiling](profiling.md).
6. For cross-framework work, use [alignment experiments](alignment.md).
7. For RCCL training, read [distributed training](distributed-training.md).
8. For model memory/throughput matrices, read
   [single-GPU benchmarking](single-gpu-benchmark.md).

## Engineering rules

- CPU reference behavior is the numerical oracle inside the engine.
- PyTorch is the independent external oracle for the supported FP32 domain.
- Readable HIP remains available when an optimized implementation is introduced.
- Optional bindings and vendor libraries do not become core dependencies.
- A benchmark result is accepted only with its environment, warm-up, repetitions,
  correctness regression, and end-to-end measurement.
- “Implemented”, “smoke-tested”, “reference-trained”, and “released” are different
  evidence states.

## Public stability

The versioned C ABI is the most stable integration boundary. The C++ API is pre-1.0
and may evolve, but public changes require shape/error tests and documentation. Python
and PyTorch integrations are optional adapters over the same engine.
