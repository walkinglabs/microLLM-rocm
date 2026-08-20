# Experiment 030 contract — BF16 activation island foundation

Status: `contract`, implementation next

## Failure being addressed

Experiment 015 cast FP32 activation at each selected Linear and kept duplicate BF16
weights. It regressed speed and memory. A valid retry must keep intermediate activations
BF16 and avoid a permanent FP32+BF16 inference copy.

## First primitive

```text
bf16_matmul(left_bf16, right_bf16, output_dtype=BFloat16)
```

Requirements:

- hipBLASLt compute remains FP32;
- output layout dtype is explicitly BF16;
- CPU reference rounds operands, accumulates FP32, then rounds output;
- no host transfer in HIP execution;
- FP32-output API remains unchanged;
- invalid dtype/device/shape/output combinations throw;
- PyTorch oracle uses BF16 inputs and BF16 output, not an FP32 approximation.

## Island gate

The first island is `gate_proj/up_proj → SwiGLU → down_proj`. It may be enabled only
after BF16-output GEMM, BF16 SwiGLU and the down projection share one device-native path.
Measure casts, Kernel calls, peak bytes and official logits. Do not add a whole-model
precision enum in this experiment.

## Falsification

If eliminating repeated casts still fails end-to-end or requires duplicate persistent
weights, discard the island and retain only the primitive. BF16 results stay on their
separate figure and never alter the FP32 score.
