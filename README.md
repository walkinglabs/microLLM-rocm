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
| CPU reference operators | smoke-tested | hand values and sanitizer tests |
| HIP readable operators | smoke-tested | 11 kernels, gfx942 conformance |
| HIP non-contiguous materialization | smoke-tested | generic rank≤8 stride-copy kernel |
| CPU Transformer autograd | smoke-tested | hand gradients and finite differences |
| SGD/AdamW | smoke-tested | hand first step and restored-state equivalence |
| Versioned checkpoint | smoke-tested | complete state, corruption, resume trajectory |
| Decoder Transformer | smoke-tested | tiny MHA/GQA forward, causal test, all-parameter backward |
| Byte tokenizer/token dataset | smoke-tested | byte round-trip and cursor-resume batches |
| BPE/token data source | loader-ready | self-contained BPE + immutable TinyStories revision |
| Tiny Transformer training | smoke-tested | 40-step overfit loss 1.81171 → 0.00673309 |
| CPU KV cache | smoke-tested | per-layer MHA/GQA cached/full logits comparison |
| Token generation | smoke-tested | greedy/top-k/temperature/fixed-seed cache generation |
| Model-S CPU forward | smoke-tested | real 15.6M model, 8192 finite logits |
| Model-S CPU training | smoke-tested | 3-step loss 11.2473 → 1.98712 |
| Model-S HIP forward | smoke-tested | MI300X CPU/HIP max logit error 4.05312e-06 |
| Tiny HIP training | smoke-tested | MI300X 5-step loss 2.21512 → 1.11681 |
| Model-M HIP train step | smoke-tested | 31.3M params; 518,798,856 peak engine bytes |
| C ABI v1 | smoke-tested | pure C CPU/HIP tensor and operator client |
| Python ctypes API | smoke-tested | dependency-free CPU/HIP integration tests |
| PyTorch Custom Ops | implemented, unverified | optional build; local Torch unavailable |
| Micro-benchmark harness | smoke-tested | CPU/HIP JSONL, Event/wall/error metadata |
| End-to-end benchmark | smoke-tested | train/generate tokens/s and engine peak memory |
| hipBLASLt + shape selector | smoke-tested | 2D FP32 correctness and Model-S measurements |
| RCCL two-GPU equivalence | smoke-tested | XGMI ranks identical; single/multi diff 1.49e-08 |
| RCCL gradient buckets | smoke-tested | 64→1 collectives: 6.676→0.225 ms |
| RCCL compute overlap | smoke-tested | separate Streams improve synthetic step 30–33% |
| RCCL four-GPU | blocked by environment | 64MB /dev/shm; failure evidence retained |
| Real-corpus Model-S/SFT report | planned | dataset/license/reference run required |
| Python/PyTorch bindings | mixed | ctypes tested; Torch source unverified |
| Profiling/autotuning | smoke-tested | M5 evidence and registry |
| Backward-ready overlap/four-GPU retry | planned | M6 follow-up |

See [STATUS.md](docs/development/STATUS.md) for the evidence gate behind each state.

Run the unified artifact/test audit with:

```bash
./scripts/verify_evidence.sh cpu
MICROLLM_BUILD_DIR=build-hip ./scripts/verify_evidence.sh hip
MICROLLM_BUILD_DIR=build-rccl ./scripts/verify_evidence.sh rccl
```

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

With HIP enabled, run the explicit-Stream CPU/HIP comparison:

```bash
./build/examples/microllm_n1_cpu_hip
```

Use the optional dependency-free Python API against a build tree:

```bash
PYTHONPATH=python \
MICROLLM_LIBRARY=build/bindings/capi/libmicrollm.so \
python3 -c 'import microllm as m; print((m.Tensor.from_f32([1,2], (2,)) * m.Tensor.from_f32([3,4], (2,))).tolist())'
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
