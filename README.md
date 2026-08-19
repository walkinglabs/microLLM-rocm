# microLLM-rocm

[![CPU evidence](https://github.com/walkinglabs/microLLM-rocm/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/walkinglabs/microLLM-rocm/actions/workflows/cpu.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://isocpp.org/)
[![ROCm](https://img.shields.io/badge/backend-ROCm%20%2F%20HIP-ED1C24.svg)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](docs/development/STATUS.md)

An independently usable C++20/HIP runtime for studying, training, profiling, and
extending small decoder-only language models on AMD GPUs.

[Documentation](docs/index.md) · [Build](docs/dev/build.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Tests](docs/dev/testing.md) ·
[Benchmarks](benchmarks/README.md) · [Roadmap](docs/development/NEXT_STEPS.md) ·
[Beginner course](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course)

> **Project maturity:** pre-alpha. The repository has measured CPU, MI300X, PyTorch
> CPU-oracle, and two-rank RCCL evidence. It does not yet claim production readiness,
> direct PyTorch ROCm parity, Radeon validation, reference-length training, or native
> Qwen/DeepSeek execution.

## Why this project exists

Large frameworks make model development productive, but they hide the ownership,
layout, execution, graph, and synchronization decisions that matter when a result is
wrong or slow. microLLM-rocm keeps those decisions visible while preserving the pieces
needed to run a real training and generation loop:

- explicit Storage/Tensor ownership, shape, stride, dtype, offset, and device;
- readable CPU references and repository-owned HIP kernels;
- an eager reverse-mode graph engine with device-native Transformer backward;
- Decoder-only MHA/GQA, RoPE, RMSNorm, SwiGLU, causal attention, loss, and optimizers;
- named model state and F32/BF16/F16 safetensors loading;
- C, Python ctypes, and optional PyTorch dispatcher adapters;
- reproducible benchmarks, rocprofv3 workflows, hipBLASLt, and RCCL experiments.

The design keeps three implementations where they provide engineering value:

```text
readable CPU reference → readable HIP kernel → measured optimized candidate
```

An optimized candidate must pass the same numerical and shape/error contracts as the
reference. A faster kernel is not accepted as a correctness argument.

## Architecture

```text
Applications / Examples / Benchmarks
                 │
       C++ API / C ABI / Python adapters
                 │
 Tensor ── Operators ── Autograd ── Transformer
   │           │             │          │
Storage     OpContext      Backward   Train / Generate
   │        Stream/Event                   │
   └──────── CPU reference / HIP runtime ──┘
                    │
             hipBLASLt / RCCL
```

Public interfaces live under `include/microllm`; implementation details stay under
`src`. Optional bindings depend on the engine, never the reverse. See the
[repository layout](docs/dev/repository-layout.md) for component ownership and
dependency invariants.

## Quick start

### CPU

Requirements: Linux, CMake 3.25+, a C++20 compiler, and Python 3.9+ for optional tests.
The current evidence was produced with CMake 3.31.10 and GCC/G++ 13.3.0.

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug
```

Run the sanitizer configuration:

```bash
cmake --preset cpu-sanitize
cmake --build --preset cpu-sanitize --parallel
ctest --preset cpu-sanitize
```

### AMD GPU

Install a ROCm release supported by the target GPU, then:

```bash
cmake --preset hip-release
cmake --build --preset hip-release --parallel
ctest --preset hip-release
```

Use an explicit architecture when auto-detection is not appropriate:

```bash
cmake --preset hip-release -DMICROLLM_HIP_ARCHITECTURES=gfx942
```

For RCCL:

```bash
cmake --preset rccl-release
cmake --build --preset rccl-release --parallel
ctest --preset rccl-release
```

The complete compiler, CMake, ROCm, library, Python, and troubleshooting matrix is in
[Build from source](docs/dev/build.md).

## Measured evidence

Current `main` gates:

| Gate | Result | Scope |
|---|---:|---|
| CPU tests | 109/109 | reference, graph, model, weights, integration |
| ASan/UBSan | 107/107 | host code; dynamic binding tests isolated |
| MI300X/gfx942 HIP | 24/24 | operators, graph, model, direct weight load |
| PyTorch CPU oracle | 2/2 | 70 FP32 numerical cases plus invalid contracts |
| Two-rank RCCL | 7/7 | collective and global-batch equivalence |
| Registered test files | 28 | machine-audited CTest registration |

Latest PyTorch-reference maximum absolute differences:

| Domain | Maximum absolute difference |
|---|---:|
| Forward operators | `1.90734863e-06` |
| Autograd graphs | `9.53674316e-07` |
| Tiny Transformer | `1.43051147e-06` |
| SGD/AdamW | `3.72529030e-08` |

These results cover the declared FP32 domain and representative shapes, not every
dtype, model size, context length, or GPU. Detailed gates are maintained in
[Testing and evidence](docs/dev/testing.md) and
[current status](docs/development/STATUS.md).

## External weights

The framework supports independent named state dictionaries, strict/non-strict model
loading, Hugging Face-style name/transpose mapping, and single or sharded safetensors:

```cpp
#include <microllm/model/model.h>

microllm::model::TransformerModel model(config);
auto mapping = microllm::model::qwen_style_weight_mapping(config);

microllm::model::LoadWeightsOptions options;
options.strict = true;
options.mapping = std::move(mapping);

model.load_safetensors_index("model.safetensors.index.json", options);
model.to(microllm::Device::hip(0));
```

The mapping API handles names and 2D linear-weight orientation. It does not implement
architecture differences such as QK-Norm, Q/K/V bias, explicit head width, MLA, MoE,
or quantization. See [Weight API](docs/WEIGHTS.md).

## Performance workflow

```bash
# Repeated Event/wall-clock micro-benchmarks
MICROLLM_BUILD_DIR=build/hip-release \
MICROLLM_BENCH_DEVICE=hip \
./scripts/run_benchmarks.sh

# HIP API, kernel, memory, JSON/CSV and Perfetto traces
./scripts/profile_hip.sh /tmp/microllm-trace -- \
  ./build/hip-release/benchmarks/microllm_bench_model \
  --mode train --model tiny --device hip \
  --steps 5 --warmup 1 --batch 1 --context 8 --new-tokens 8
```

The current exact-shape registry covers readable 2D matmul and hipBLASLt. It is not a
general autotuner, and there is no stable in-process `@profile` API yet. The supported
workflow and missing profiler boundary are documented in [Profiling](docs/dev/profiling.md)
and [Operator development](docs/dev/operator-development.md).

## Repository map

| Path | Responsibility |
|---|---|
| `include/microllm/` | public C++ and C APIs |
| `src/` | runtime, Tensor, operators, autograd, model, IO, train/infer, RCCL |
| `bindings/` | optional C, Python, and PyTorch adapters |
| `apps/` | command-line applications |
| `examples/` | small executable API examples |
| `benchmarks/` | micro/e2e/distributed benchmarks and curated evidence |
| `tests/` | unit, graph, conformance, integration, and coverage gates |
| `docs/` | framework and developer documentation |
| `scripts/` | reproducible build, test, benchmark, and profile workflows |

## Documentation

- [Documentation index](docs/index.md)
- [Developer guide](docs/dev/index.md)
- [Build system and validated toolchains](docs/dev/build.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Operator contracts](docs/OPERATOR_CONTRACTS.zh-CN.md)
- [Weights and safetensors](docs/WEIGHTS.md)
- [Hardware compatibility](docs/COMPATIBILITY.md)
- [Current evidence status](docs/development/STATUS.md)
- [Roadmap and explicit gaps](docs/development/NEXT_STEPS.md)
- [Chronological development records](docs/development/README.md)

The beginner course is maintained separately on
[`tutorial/beginner-course`](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course).

## Contributing

Changes require an explicit contract, a reference, positive and negative tests, and
reproducible evidence. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/TASK_CONTRACT.md](docs/TASK_CONTRACT.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
