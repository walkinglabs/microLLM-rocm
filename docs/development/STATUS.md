# Evidence status

States: `draft`, `implemented`, `smoke-tested`, `reference-trained`, `released`.

| Component | State | Current evidence | Missing gate |
|---|---|---|---|
| CPU configuration | smoke-tested | framework-only main configure/build; 109/109 CPU CTest | CI matrix |
| Device/DType | smoke-tested | unit and invalid-index tests | HIP runtime use |
| CPU Storage | smoke-tested | sharing/lifetime/zero-byte tests | sanitizer log in CI |
| Tensor metadata/views | smoke-tested | hand values, randomized shapes, bounds | more dtypes |
| HIP view materialization | smoke-tested | gfx942 transposed logical-order copy | rank>8/more dtypes |
| Tensor PPM sample | smoke-tested | executable output/checksum | documented golden value |
| HIP Storage/runtime | smoke-tested | gfx942 allocation, transfer, Stream/Event tests | CI on more GPUs |
| CPU reference operators | smoke-tested | hand values plus PyTorch forward/backward oracle for every public math op | more dtypes |
| HIP readable operators | smoke-tested | 24/24 gfx942 suite; ops, graph, and direct weight load | more architectures/optimized paths |
| Operator context | smoke-tested | explicit Stream ordering and mismatch tests | low-level C descriptor |
| CPU Transformer Autograd | smoke-tested | dedicated graph construction tests, finite differences, PyTorch full-graph gradients | more dtypes |
| HIP Autograd | smoke-tested | CPU/HIP full Transformer gradient comparison; zero host transfers during graph execution | optimized reductions/more dtypes |
| SGD/AdamW | smoke-tested | PyTorch SGD and two-step AdamW parameter/moment parity plus restored next step | device-native HIP update/mixed precision |
| Checkpoint | smoke-tested | atomic complete-state load, corruption, 3-step resume | mixed precision |
| Model-S/Model-M config | smoke-tested | executable exact parameter/byte tests | model layers/training |
| Model-S CPU forward | smoke-tested | 15,586,176 parameters and 8192 finite logits | training/HIP |
| Model-S CPU training | smoke-tested | 3-step loss trajectory, AdamW state, parameter delta | real corpus/HIP |
| Model-S HIP forward | smoke-tested | MI300X/gfx942 8192-logit CPU comparison | multi-token/preallocated cache |
| Tiny HIP training | smoke-tested | 5-step finite loss/grad trajectory on MI300X | device-native AdamW |
| Model-M HIP train step | smoke-tested | 31.3M params, finite backward/update, 518.8MB engine peak | multi-step/real corpus |
| Decoder Transformer structure | smoke-tested | tiny GQA graph topology, PyTorch logits/loss/all-gradient parity, CPU/HIP parity | overfit/full recipe |
| Byte tokenizer/token dataset | smoke-tested | all-byte round-trip and cursor equivalence | real-corpus run |
| BPE/TinyStories source | smoke-tested | BPE round-trip + immutable licensed range + Model-S smoke | full corpus/reference train |
| Weight/state API | smoke-tested | independent state_dict, atomic strict/non-strict load, Qwen-style transpose mapping | streaming load/model-specific architecture validation |
| safetensors | smoke-tested | F32/BF16/F16, single/sharded/index files, corruption and CPU/HIP target tests | FP8/quantized formats/memory mapping |
| C++ training CLI | smoke-tested | CPU save/resume fixture and Model-S HIP real-text steps | validation/report CLI |
| Model-S TinyStories smoke | smoke-tested | immutable 1MiB train prefix, 10 HIP steps | full train/validation curve |
| Tiny Transformer training | smoke-tested | 40-step overfit and finite gradients | validation split/Model-S |
| SFT response masking | smoke-tested | CPU/HIP ignored targets and tiny response-loss curve | Model-S instruction corpus/run |
| CPU KV cache | smoke-tested | every prefix, MHA/GQA logits tolerance | preallocation/HIP/batching |
| Token generation | smoke-tested | deterministic sampling and cache-backed length/bounds | trained text report |
| Stable model failure | smoke-tested | low-loss cycle breaks beyond training context | rebuttal experiments |
| PyTorch Custom Ops | smoke-tested on CPU | Torch 2.13 add/multiply via dispatcher | build/run with PyTorch ROCm |
| PyTorch correctness oracle | smoke-tested on CPU | 30 Tensor APIs, 29 graph APIs, 25 test files audited; forward/backward/shape/model/optimizer parity | direct PyTorch ROCm run |
| PyTorch ROCm environment | draft | matching 2.11/7.13 wheel Bus error on import | working Torch ROCm environment |
| C ABI v1 | smoke-tested | pure C CPU/HIP create/copy/ops/error client | zero-copy external views |
| Python ctypes API | smoke-tested | CPU/HIP Tensor/ops/error unittest | packaging/broader ops |
| External TensorView ops | smoke-tested | caller-owned CPU/HIP buffers and Stream | Torch build validation |
| Profiling/autotune | smoke-tested | rocprofv3, hipBLASLt, shape registry | persistent arch/version cache |
| Micro-benchmark harness | smoke-tested | CPU/HIP Event+wall JSONL and error gate | PyTorch comparison/more shapes |
| Engine allocation tracker | smoke-tested | CPU/HIP current/peak/total accounting | external allocator integration |
| End-to-end benchmark | smoke-tested | CPU/gfx942 train+generate raw JSONL | Model-S/PyTorch/tuned comparison |
| rocprofv3 workflow | smoke-tested | kernel/HIP API/memory/full trace generated | release artifact retention |
| hipBLASLt matmul | smoke-tested | FP32 CPU comparison, shape matrix, Model-S e2e | batched/workspace/autotune cache |
| Matmul tuning registry | smoke-tested | exact shape override/clear and availability gates | persistence/arch key |
| RCCL two-GPU baseline | smoke-tested | XGMI average and global-batch parameter equivalence | buckets/4 GPU/failure timing |
| RCCL gradient buckets | smoke-tested | 1MB payload with 64/4/1 bucket matrix | overlap with backward |
| RCCL four-GPU | draft | 3 stable init failures and debug root cause | environment with >87MB /dev/shm |
| RCCL compute overlap | smoke-tested | 3 runs, 30–33% synthetic overlap gain | bucket readiness during real backward |
