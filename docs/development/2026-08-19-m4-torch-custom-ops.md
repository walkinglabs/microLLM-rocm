# 2026-08-19 — M4 optional PyTorch Custom Ops

## Contract

Register microLLM add/multiply under `torch.ops.microllm` without making PyTorch a
core dependency. PyTorch retains Tensor allocation and ownership. ROCm execution must
launch on PyTorch's current HIP stream through the external OpContext seam.

## Implementation

- `MICROLLM_BUILD_TORCH_OPS=AUTO|ON|OFF`;
- `find_package(Torch)` only in the optional binding directory;
- `TORCH_LIBRARY` schema plus CPU and CUDA-dispatch implementations (PyTorch ROCm uses
  the CUDA dispatch/device API surface);
- zero-copy TensorView mapping from PyTorch sizes, strides, pointers, and device;
- output allocated by `torch::empty_like`;
- PyTorch current HIP stream passed as a non-owning external stream;
- Python loader and CPU/ROCm comparison tests enabled only when Torch is found.

The implementation follows the official PyTorch C++ Custom Operators pattern:
<https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html>.

## Evidence state

The base environment initially had neither the `torch` Python package nor
`TorchConfig.cmake`; `AUTO` correctly left the target disabled. Isolated CPU and ROCm
environments now compile the adapter. The measured ROCm environment is Torch
`2.11.0+rocm7.13.0rc2`, HIP `7.13.99004`, `gfx942`.

The adapter maps FP32/FP16/BF16 PyTorch allocations without copying, launches on the
current HIP Stream, registers explicit add/multiply Autograd formulas and supplies a
Meta implementation for `torch.compile(fullgraph=True)`. Six fresh processes cover 20
forward and forward/backward cases with complete Max/RMS/loss zero. This is an
integration result, not a speed claim: every Event median is below native Torch
(`0.469×–0.973×` as Torch/microLLM), with identical allocator peaks. See
[Experiment 329](../optimization-log/experiments/329-pytorch-rocm-custom-ops.md).
