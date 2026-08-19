# microLLM-rocm

`microLLM-rocm` is a small, independently usable C++/HIP training and inference
engine for teaching, measuring, and improving language-model systems on AMD GPUs.

The project is **pre-alpha**. The CPU float32 Storage/Tensor path and its N0 example
are implemented. HIP operators, autograd, Transformer training, Python/PyTorch
bindings, profiling, and distributed execution are tracked as explicit milestones;
they are not presented as completed features.

## Product shape

```text
Optional Python API / PyTorch Custom Ops
                    ↓
        C API and C++ engine API
                    ↓
 Tensor → Ops → Autograd → Model → Train/Infer
                    ↓
       CPU reference / AMD HIP runtime
                    ↓
        hipBLASLt / RCCL when enabled
```

Every performance-sensitive operator will keep three paths where useful:

1. a readable CPU reference;
2. a readable HIP implementation;
3. a tuned implementation, which may use ROCm vendor libraries.

## Current status

| Area | Status | Evidence |
|---|---|---|
| CPU build without ROCm | smoke-tested | CPU CMake preset and CTest |
| Device/Storage | smoke-tested | ownership and lifetime tests |
| Tensor shape/stride/view | smoke-tested | deterministic and randomized tests |
| N0 PPM example | smoke-tested | runnable example with checksum |
| HIP runtime and Tensor transfer | smoke-tested | MI300X runtime tests |
| CPU/HIP operators | planned | M1 next step |
| Autograd and checkpointing | planned | M2 |
| Model-S training/inference | planned | M3 |
| Python/PyTorch bindings | planned | M4 |
| Profiling/autotuning | planned | M5 |
| RCCL multi-GPU | planned | M6 |

See [STATUS.md](docs/development/STATUS.md) for the evidence gate behind each state.

## Build and test

CPU-only development does not require ROCm:

```bash
./scripts/configure.sh -DMICROLLM_ENABLE_HIP=OFF
./scripts/build.sh
./scripts/test.sh
```

Run the stricter host check with AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
./scripts/check_cpu.sh
```

On a ROCm machine, `MICROLLM_ENABLE_HIP=AUTO` is the default. Use `ON` when a
missing HIP toolchain should be a configuration error.

Run the first end-to-end artifact:

```bash
./build/examples/microllm_n0_ppm /tmp/microllm-n0.ppm
```

## Models

- **Model-S:** approximately 15.6M parameters / 62 MB of FP32 weights.
- **Model-M:** approximately 31M parameters / 124 MB of FP32 weights (planned).

Reports always state both parameter count and weight dtype. “64 MB model” is not
used as a substitute for a parameter count.

## Documentation

- [Project charter](docs/PROJECT_CHARTER.md)
- [Architecture and dependency rules](docs/ARCHITECTURE.md)
- [Development roadmap](docs/development/ROADMAP.md)
- [Development records](docs/development/README.md)
- [Contributor task contract](docs/TASK_CONTRACT.md)
- [Contributing](CONTRIBUTING.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
