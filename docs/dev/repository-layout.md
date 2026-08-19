# Repository layout and dependency rules

The layout follows a component model: public headers, implementations, tests, adapters,
tools, and evidence have distinct ownership.

```text
microLLM-rocm/
├── include/microllm/   public C++ and C interfaces
├── src/                engine component implementations
├── bindings/           optional C, Python, and PyTorch adapters
├── apps/               end-user command-line applications
├── examples/           small runnable API examples
├── benchmarks/         micro/e2e/distributed benchmarks and curated results
├── tests/              unit, graph, conformance, and integration tests
├── python/             optional Python package and Python-side tests
├── docs/               framework, developer, and evidence documentation
├── scripts/            reproducible build/test/profile helpers
├── data/               dataset registry and generated smoke fixtures
├── cmake/              project CMake modules when component logic is shared
├── CMakeLists.txt      root build definition
└── CMakePresets.json   supported developer build entry points
```

## Engine components

```text
base
  ↓
runtime → core → ops → autograd → model → training/inference
                                      ↘ multi_gpu (optional RCCL)
io ───────────────────────────────────↗
```

- `runtime`: allocation, copy, device, Stream, Event, and counters.
- `core`: Storage, Tensor metadata, views, and device transfer.
- `ops`: CPU references, readable HIP, optimized candidates, and execution context.
- `autograd`: eager reverse-mode graph construction and traversal.
- `model`: Decoder-only Transformer composition and named weight loading.
- `io`: token data, BPE, SFT batches, and safetensors.
- `training`: optimizers, checkpoint, and training step.
- `inference`: sampling, generation, and KV cache.
- `multi_gpu`: RCCL communicator and gradient buckets.

## Dependency invariants

- `base` uses only the C++ standard library.
- CPU `runtime/core/ops` build without ROCm headers or libraries.
- `ops` never depends on model or training code.
- the CPU reference remains available when a HIP/vendor implementation is added.
- bindings depend on the engine; the engine never depends on a binding.
- RCCL is compiled only when explicitly enabled.
- public C headers do not expose C++ or HIP implementation types.
- development evidence may describe failed experiments but cannot expand support claims.

The beginner course is deliberately absent from `main`; it lives on
`tutorial/beginner-course` and consumes the framework as a downstream deliverable.
