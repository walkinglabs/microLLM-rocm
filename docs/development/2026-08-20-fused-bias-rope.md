# 2026-08-20 — fused projection bias and split-half RoPE

## Framework change

Qwen-style attention used to allocate and launch once for Q/K bias, then again for
split-half RoPE. The new FP32 operator consumes `[B,H,T,D]` projection values plus a
`[H*D]` bias and writes the rotated result directly.

The model selects it only when the attention projection has bias and the configured
layout is split-half. Bias-free and interleaved-RoPE models keep the existing path. V
projection bias is deliberately unchanged.

## Training contract

`autograd::rope_split_half_bias` is a first-class graph operation, not an inference-only
shortcut. Its backward:

1. applies inverse split-half RoPE to the upstream gradient;
2. sends that FP32 gradient to the projection input;
3. restores `[B,T,H*D]` layout and reduces B/T into the bias gradient.

The independent PyTorch oracle builds `rope(x + bias.view(1,H,1,D))` and compares the
forward value, input gradient and bias gradient.

## Measured result

Three baseline and three candidate processes used the official Qwen/DeepSeek matrix,
2 warm-ups and 5 measured steps:

```text
Qwen generation paired median       +13.7%
DeepSeek generation paired median    +6.6%
Qwen / DeepSeek training             +0.1% / +0.4%
fixed PyTorch-reference score         1.784147
```

The matched DeepSeek profile removed exactly 1,120 Kernel launches and lowered
`hipLaunchKernel` API duration from 77.80 to 65.26 ms. Engine peak bytes did not increase.

## Verification

```text
CPU debug                         156/156 pass
CPU ASan/UBSan                    154/154 pass
MI300X/gfx942 HIP                  55/55 pass
Python/PyTorch operator+graph       4/4 pass
```

Full raw evidence and the paired-median caveat are in
`docs/optimization-log/experiments/016-fused-bias-rope.md`.
