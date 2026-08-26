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

A follow-up packet experiment rejects universal vectorization because FP32 regresses.
The retained Auto route is deliberately shape- and dtype-selective: aligned FP16/BF16
tensors at or above 4,194,304 elements use 16-byte packets, while FP32, smaller tensors,
and misaligned views remain scalar. The four 16M low-precision rows improve
`1.277×–1.411×` versus the scalar adapter with exact outputs and unchanged peaks. See
[Experiment 330](../optimization-log/experiments/330-pytorch-custom-op-vector16.md).

The adapter's first graph-reducing operation is SwiGLU. It replaces PyTorch SiLU plus
multiply and their intermediate allocation with one caller-owned microLLM output. The
schema includes Meta and Autograd; FP32 backward calls the fused caller-owned engine
primitive, while FP16/BF16 retain explicit readable formulas. At 16M elements, forward
is `1.142×–1.570×` native Torch and measured peak halves. Forward+backward remains only
`0.615×–0.761×`, so the adapter does not make a training speed claim. See
[Experiment 331](../optimization-log/experiments/331-pytorch-custom-op-swiglu.md).

The first backward rebuttal rejects and removes a float4 candidate: its Event ratio is
only `0.946×–1.039×` versus the retained scalar producer. The scalar producer itself is
already `2.07×–2.82×` the readable native formula with lower peak. Training work therefore
moves to the expanded scalar seed that the Python Autograd bridge currently materializes,
not another packet-width search. See
[Experiment 332](../optimization-log/experiments/332-pytorch-swiglu-backward-vector-reject.md).

The next layout audit proves that `sum()` produces a four-byte, zero-stride expanded
gradient. The bridge now sends a one-element device view to a scalar-seed fused backward
only when every gradient stride is zero. Mean, weighted and general gradients retain the
ordinary path. This removes 99.42%–99.96% of measured temporary peak and improves 64K/1M
F+B by `1.164×/1.081×`, but remains below native Torch. See
[Experiment 333](../optimization-log/experiments/333-pytorch-swiglu-scalar-seed.md).
