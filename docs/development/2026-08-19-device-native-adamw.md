# 2026-08-19 — device-native AdamW and real Qwen training step

## Problem found

The first AdamW implementation copied parameters, gradients, and both moment tensors to
the host for every step. It was numerically correct on tiny tests but would move many
gigabytes across PCIe for a Hugging Face model.

## Change

- added an in-place FP32 AdamW CPU reference operator;
- added a one-thread-per-element HIP AdamW Kernel;
- parameters, gradients, first moments, and second moments stay on one device;
- optimizer retains bias correction and decoupled weight decay semantics;
- CPU hand/PyTorch optimizer tests remain unchanged;
- MI300X test requires zero H2D and zero D2H calls during `step()`.

## Official Qwen2.5-0.5B training evidence

```text
microLLM loss             6.836029530
PyTorch FP32 loss         6.836015701
loss absolute difference  1.383e-5
final_norm[0] before      7.468750000
microLLM after            7.468739033
PyTorch after             7.468739033
MI300X full step           2267.60 ms
MI300X AdamW update        10.68 ms
optimizer host transfers  0 / 0
engine peak               9.557 GB
```

This proves one real step. It does not yet provide a multi-step Qwen training curve,
validation corpus, BF16 mixed precision, or checkpoint-resume experiment.
