# Evidence status

States: `draft`, `implemented`, `smoke-tested`, `reference-trained`, `released`.

| Component | State | Current evidence | Missing gate |
|---|---|---|---|
| CPU configuration | smoke-tested | clean CPU configure/build/CTest | CI matrix |
| Device/DType | smoke-tested | unit and invalid-index tests | HIP runtime use |
| CPU Storage | smoke-tested | sharing/lifetime/zero-byte tests | sanitizer log in CI |
| Tensor metadata/views | smoke-tested | hand values, randomized shapes, bounds | more dtypes |
| N0 PPM | smoke-tested | executable output/checksum | documented golden value |
| HIP Storage/runtime | smoke-tested | gfx942 allocation, transfer, Stream/Event tests | CI on more GPUs |
| CPU reference operators | smoke-tested | 9 hand-value/stability/shape tests | gradient references |
| HIP readable operators | smoke-tested | gfx942 conformance for 11 kernels | more architectures/optimized paths |
| Operator context | smoke-tested | explicit Stream ordering and mismatch tests | low-level C descriptor |
| CPU Transformer Autograd | smoke-tested | 11 focused tests and finite differences | attention composition/GPU backward |
| SGD/AdamW | smoke-tested | hand first step and state-restored next step | GPU update/mixed precision |
| Checkpoint | smoke-tested | complete-state load, corruption, 3-step resume | atomic save/GPU tensors |
| Model-S/Model-M config | smoke-tested | executable exact parameter/byte tests | model layers/training |
| Decoder Transformer structure | smoke-tested | tiny GQA/MHA causal forward and all-parameter backward | overfit/full recipe |
| Byte tokenizer/token dataset | smoke-tested | all-byte round-trip and cursor equivalence | BPE/real corpus |
| Tiny Transformer training | smoke-tested | 40-step overfit and finite gradients | validation split/Model-S |
| Python/PyTorch bridge | draft | ABI design | zero-copy implementation |
| Profiling/autotune | draft | registry design | traces and selection cache |
| RCCL | draft | roadmap | 2/4 GPU equivalence |
