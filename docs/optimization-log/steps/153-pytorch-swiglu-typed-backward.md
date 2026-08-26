# Step 153 — typed fused low-precision SwiGLU backward

Status: complete; keep, scoped line closed

FP16/BF16两份梯度由一个HIP Kernel生成。相对C++ ATen公式提升1.257×–1.319×，相对native
Torch为1.048×–1.084×，peak完全相同；BF16 bit-exact，FP16 Max2.38e-7。SwiGLU adapter线关闭。

详细记录见[Experiment 337](../experiments/337-pytorch-swiglu-typed-backward.md)。

