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
`TorchConfig.cmake`; `AUTO` correctly left the target disabled. A later isolated
official Torch 2.13 CPU environment compiled the dispatcher library and passed CPU
add/multiply comparison. The PyTorch ROCm case remains unverified, so current-stream
HIP integration must not yet be presented as measured.
