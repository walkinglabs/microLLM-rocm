# Evidence status

States: `draft`, `implemented`, `smoke-tested`, `reference-trained`, `released`.

| Component | State | Current evidence | Missing gate |
|---|---|---|---|
| CPU configuration | smoke-tested | clean CPU configure/build/CTest | CI matrix |
| Device/DType | smoke-tested | unit and invalid-index tests | HIP runtime use |
| CPU Storage | smoke-tested | sharing/lifetime/zero-byte tests | sanitizer log in CI |
| Tensor metadata/views | smoke-tested | hand values, randomized shapes, bounds | more dtypes |
| HIP view materialization | smoke-tested | gfx942 transposed logical-order copy | rank>8/more dtypes |
| N0 PPM | smoke-tested | executable output/checksum | documented golden value |
| HIP Storage/runtime | smoke-tested | gfx942 allocation, transfer, Stream/Event tests | CI on more GPUs |
| CPU reference operators | smoke-tested | 9 hand-value/stability/shape tests | gradient references |
| HIP readable operators | smoke-tested | gfx942 conformance for 11 kernels | more architectures/optimized paths |
| Operator context | smoke-tested | explicit Stream ordering and mismatch tests | low-level C descriptor |
| CPU Transformer Autograd | smoke-tested | 11 focused tests and finite differences | attention composition/GPU backward |
| SGD/AdamW | smoke-tested | hand first step and state-restored next step | GPU update/mixed precision |
| Checkpoint | smoke-tested | complete-state load, corruption, 3-step resume | atomic save/GPU tensors |
| Model-S/Model-M config | smoke-tested | executable exact parameter/byte tests | model layers/training |
| Model-S CPU forward | smoke-tested | 15,586,176 parameters and 8192 finite logits | training/HIP |
| Model-S CPU training | smoke-tested | 3-step loss trajectory, AdamW state, parameter delta | real corpus/HIP |
| Model-S HIP forward | smoke-tested | MI300X/gfx942 8192-logit CPU comparison | multi-token/preallocated cache |
| Tiny HIP training | smoke-tested | 5-step finite loss/grad trajectory on MI300X | device-native backward/AdamW |
| Decoder Transformer structure | smoke-tested | tiny GQA/MHA causal forward and all-parameter backward | overfit/full recipe |
| Byte tokenizer/token dataset | smoke-tested | all-byte round-trip and cursor equivalence | BPE/real corpus |
| Tiny Transformer training | smoke-tested | 40-step overfit and finite gradients | validation split/Model-S |
| CPU KV cache | smoke-tested | every prefix, MHA/GQA logits tolerance | preallocation/HIP/batching |
| Token generation | smoke-tested | deterministic sampling and cache-backed length/bounds | trained text report |
| Stable model failure | smoke-tested | low-loss cycle breaks beyond training context | rebuttal experiments |
| PyTorch Custom Ops | implemented | optional registration source, TensorView seam | build/run with PyTorch ROCm |
| C ABI v1 | smoke-tested | pure C CPU/HIP create/copy/ops/error client | zero-copy external views |
| Python ctypes API | smoke-tested | CPU/HIP Tensor/ops/error unittest | packaging/broader ops |
| External TensorView ops | smoke-tested | caller-owned CPU/HIP buffers and Stream | Torch build validation |
| Profiling/autotune | draft | registry design | traces and selection cache |
| Micro-benchmark harness | smoke-tested | CPU/HIP Event+wall JSONL and error gate | PyTorch comparison/more shapes |
| Engine allocation tracker | smoke-tested | CPU/HIP current/peak/total accounting | external allocator integration |
| End-to-end benchmark | smoke-tested | CPU/gfx942 train+generate raw JSONL | Model-S/PyTorch/tuned comparison |
| rocprofv3 workflow | smoke-tested | kernel/HIP API/memory/full trace generated | release artifact retention |
| RCCL | draft | roadmap | 2/4 GPU equivalence |
