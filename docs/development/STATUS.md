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
| HIP basic operators | smoke-tested | gfx942 fill/elementwise/matmul conformance | more architectures |
| HIP Transformer operators | draft | CPU oracle | GPU kernels/conformance |
| Autograd | draft | roadmap | finite-difference tests |
| Checkpoint | draft | roadmap | multi-step resume |
| Model-S | draft | corrected parameter budget | overfit and training run |
| Python/PyTorch bridge | draft | ABI design | zero-copy implementation |
| Profiling/autotune | draft | registry design | traces and selection cache |
| RCCL | draft | roadmap | 2/4 GPU equivalence |
