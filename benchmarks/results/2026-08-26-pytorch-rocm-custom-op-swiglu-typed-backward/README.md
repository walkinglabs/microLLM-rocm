# Typed fused FP16/BF16 SwiGLU backward

This matrix replaces the C++ Autograd low-precision ATen formula with one caller-owned HIP Kernel.
The Kernel loads FP16/BF16 gate, up and output gradient, evaluates in FP32, then rounds each of the
two outputs once to the input dtype. CPU and HIP references use the same contract.

Six fresh processes repeat the full 15-case SwiGLU matrix. At 64K/1M, BF16 F+B is
`1.084×/1.055×` native Torch and FP16 is `1.074×/1.048×`; measured peak equals native in all four
rows. BF16 gradients are bit-exact to the PyTorch oracle; FP16 maximum error is `2.38e-7`.

`comparison.json` records the improvement over the prior C++ ATen formula and all admission gates.
FP32 keeps its independently accepted scalar/general producers unchanged.

