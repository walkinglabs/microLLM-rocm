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
[Optimization log](docs/optimization-log/README.md) ·
[Beginner course](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course)

> **Project maturity:** pre-alpha. The repository has measured CPU, MI300X, PyTorch
> CPU-oracle, and two-rank RCCL evidence. It does not yet claim production readiness,
> all-workload PyTorch ROCm parity, Radeon validation, or reference-length training.

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
- MI300X FNUZ FP8 quantize/dequantize, scaled hipBLASLt GEMM, FP32-master Transformer
  training policy, and KV-cache decode;
- single-representation BF16 FFN/Attention projection inference for pinned Qwen/DeepSeek,
  with shared QKV cast, exact-token, memory, throughput and PyTorch BF16 evidence;
- C, Python ctypes, and optional PyTorch dispatcher adapters;
- reproducible benchmarks, rocprofv3 workflows, hipBLASLt, and RCCL experiments.
- a cross-framework trace runner for operator/layer values and latency comparisons.

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
| CPU tests | 163/163 | reference, Qwen/DeepSeek, graph/model/weights, benchmark, PyTorch and optimization-log schemas |
| ASan/UBSan | 161/161 | host code; dynamic binding tests isolated |
| MI300X/gfx942 HIP | 63/63 | paired KV store, allocator Events, fused ops, BF16/FP8 and model matrix |
| PyTorch CPU operator/model oracle | 4/4 | forward/backward/shape, mixed BF16 FFN model and optimizer parity |
| Two-rank RCCL | 11/11 | collectives, global-batch equivalence, DDP trainer/CLI |
| Registered test files | 34 | machine-audited CTest registration |
| CPU source coverage | 83.9% lines / 66.6% branches | GCC 13.3 + gcovr 8.3; `src/` and `include/` |

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

Latest single-MI300X FP32 model matrix. Built-in rows are CI smoke measurements;
official rows exclude two warm-up iterations and measure five iterations:

| Model | Mode | Measured throughput | Peak engine memory |
|---|---|---:|---:|
| Model-S, 15.6M | train / generate | 1.111 / 1.217 token/s | 238.687 / 59.608 MiB |
| Model-M, 31.3M | train / generate | 0.528 / 1.226 token/s | 478.765 / 119.754 MiB |
| Qwen2.5-0.5B official | train / generate | 24.027 / 18.847 token/s | 8.901 / 2.349 GiB |
| DeepSeek Distill Qwen 1.5B official | train / generate | 13.295 / 10.053 token/s | 26.514 / 6.622 GiB |

These are short functional measurements with random built-in models or fixed official
prompts, not long-context or stable serving claims. “Peak engine memory” excludes
driver/vendor-private allocations. Commands and raw JSONL are documented in
[single-GPU benchmarking](docs/dev/single-gpu-benchmark.md).

Matched Python/PyTorch ROCm comparison on the same MI300X:

| Model | Mode | microLLM | PyTorch | microLLM/PyTorch |
|---|---|---:|---:|---:|
| Model-S | train / generate | 13.57 / 139.22 token/s | 177.57 / 293.55 token/s | 0.076× / 0.474× |
| Model-M | train / generate | 3.51 / 90.57 token/s | 59.94 / 237.60 token/s | 0.059× / 0.381× |
| Qwen2.5-0.5B | train / generate | 24.03 / 18.85 token/s | 51.32 / 70.18 token/s | 0.468× / 0.269× |
| DeepSeek Distill Qwen 1.5B | train / generate | 13.30 / 10.05 token/s | 26.23 / 62.40 token/s | 0.507× / 0.161× |

All comparison rows use matched warm-up/repetition settings and exclude warm-up from
reported throughput. See [Python/PyTorch comparison](docs/dev/pytorch-benchmark.md) for
raw data, memory ratios, implementation differences and limitations.

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

Run the same model in microLLM and PyTorch, then compare every recorded value, shape,
operator time, layer time, and full forward time:

```bash
python3 tools/alignment/run.py \
  --microllm-binary build/hip-release/apps/microllm_alignment \
  --python /path/to/python-with-pytorch \
  --output /tmp/microllm-alignment \
  --microllm-device hip \
  --pytorch-device cpu \
  --warmup 5 --repetitions 20
```

See [Alignment experiments](docs/dev/alignment.md) for the trace schema, four-pass
measurement design, comparison metrics, artifact manifest, and model-extension process.

Inspect the pinned Qwen2.5-compatible architecture without allocating model weights:

```bash
build/cpu-debug/apps/microllm_hf_inspect \
  --config tests/fixtures/qwen25-0.5b-config.json
```

The official Qwen2.5-0.5B checkpoint now passes complete-logit and greedy KV-cache
comparison on MI300X. See the commands, metrics, remaining chat/BF16 gates, and honest
scope in [the Qwen2.5 development record](docs/development/2026-08-19-qwen25-architecture.md).

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
general autotuner. The C++ `TraceSession`/`TraceTimer` API is implemented; a Python
`@profile` decorator and asynchronous rocprof range correlation remain future work.
See [Profiling](docs/dev/profiling.md) and
[Operator development](docs/dev/operator-development.md).

## Multi-GPU training

The RCCL build includes a correctness-first `DataParallelTrainer` and CLI:

```bash
./build/rccl-release/apps/microllm_distributed_train \
  --steps 3 --bucket-bytes 4194304 \
  --trace /tmp/microllm-ddp-trace.jsonl
```

It runs rank-local forward/backward, bucketed average all-reduce, identical AdamW
updates, cross-rank parameter checks, and stage-level profiling. The current baseline
synchronizes backward before communication; it does not yet claim gradient-ready
overlap or one-process-per-GPU production semantics. See
[Distributed training](docs/dev/distributed-training.md).

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
| `tools/alignment/` | microLLM/PyTorch run orchestration and comparison reports |

## Documentation

- [Documentation index](docs/index.md)
- [Developer guide](docs/dev/index.md)
- [Build system and validated toolchains](docs/dev/build.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Operator contracts](docs/OPERATOR_CONTRACTS.zh-CN.md)
- [Weights and safetensors](docs/WEIGHTS.md)
- [Tensor dtypes and MI300/MI350 precision policy](docs/DTYPES.md)
- [Hugging Face and verified Qwen2.5 workflow](docs/HUGGINGFACE.md)
- [DeepSeek Distill support and flagship boundary](docs/DEEPSEEK.md)
- [Hardware compatibility](docs/COMPATIBILITY.md)
- [Alignment experiments](docs/dev/alignment.md)
- [Distributed training](docs/dev/distributed-training.md)
- [Current evidence status](docs/development/STATUS.md)
- [Roadmap and explicit gaps](docs/development/NEXT_STEPS.md)
- [Chronological development records](docs/development/README.md)
- [Living 0→1 optimization blog and experiment log](docs/optimization-log/README.md)

The course-only N0–N10 curriculum is maintained separately on
[`tutorial/beginner-course`](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course).
That branch contains teaching documents and assignments, not a copy of this engine.

## Contributing

Changes require an explicit contract, a reference, positive and negative tests, and
reproducible evidence. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/TASK_CONTRACT.md](docs/TASK_CONTRACT.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
