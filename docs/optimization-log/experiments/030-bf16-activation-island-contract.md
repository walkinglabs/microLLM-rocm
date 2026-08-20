# Experiment 030 contract — BF16 activation island foundation

Status: `operator island kept` — whole-model integration is still pending

## Failure being addressed

Experiment 015 cast FP32 activation at each selected Linear and kept duplicate BF16
weights. It regressed speed and memory. A valid retry must keep intermediate activations
BF16 and avoid a permanent FP32+BF16 inference copy.

## First primitive

```text
bf16_matmul(left_bf16, right_bf16, output_dtype=BFloat16)
```

Implemented in commit `85438ed`. CPU rounded reference and MI300X hipBLASLt BF16-output
paths pass focused dtype/value/zero-transfer and coverage gates. The legacy FP32-output
entry point is unchanged.

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

## What was implemented

The public `ops::bf16_ffn` contract is:

```text
FP32 input
  ↓ cast once
BF16 gate GEMM ─┐
BF16 up GEMM   ─┴→ BF16 SwiGLU → BF16 down GEMM
                                      ↓
                                  FP32 output
```

All three weights arrive as BF16. The operator does not create or retain an extra weight
copy. The caller therefore owns the future policy decision: load inference weights as
BF16, or keep FP32 master weights for training. Input/output use FP32 only at the residual
boundary.

MI300X exposed a real-shape exception that the 128-square smoke missed. Qwen decode
BF16-input GEMM with direct FP32 output returned hipBLASLt status 6 for several small-M
shapes. The retained implementation probes once per `(M,K,N)` shape. If direct FP32 output
is rejected, it uses BF16 output plus a device cast and remembers that capability result;
it does not retry the failed library path in every layer.

## Evidence matrix

Environment: AMD Instinct MI300X VF, `gfx942:sramecc+:xnack-`, HIP runtime `71399004`.
Each number is the median of three independent process medians; each process uses five
warm-ups and twenty measured iterations. Timing uses GPU Events, not an unsynchronized
CPU timer.

| Shape | Island ms | vs FP32 | vs per-Linear BF16 | Relative L2 vs FP32 |
|---|---:|---:|---:|---:|
| Qwen, M=1, 896→4864→896 | 0.057370 | 1.232× | 1.067× | 3.20% |
| Qwen, M=128, 896→4864→896 | 0.057050 | 1.392× | 1.081× | 2.60% |
| DeepSeek, M=1, 1536→8960→1536 | 0.083189 | 1.117× | 1.088× | 1.62% |
| DeepSeek, M=128, 1536→8960→1536 | 0.103737 | 1.576× | 1.091× | 1.25% |

All 36 measured records passed the accuracy gate and had zero H2D/D2H payload calls in
the measured region. At M=128, peak active engine bytes fell from 88,997,888 to 84,246,528
for Qwen and from 266,928,128 to 258,146,304 for DeepSeek versus per-Linear BF16.

![BF16 FFN activation island](../assets/bf16-ffn-island.svg)

The profiler run is deliberately not used for speed: instrumentation inflated execution
time. It is used only for structure. With the common setup/reference dispatches removed,
the Qwen M=1 path had eight dispatches for per-Linear BF16 and six for the island. The
FP32→BF16 input casts fell from three to one; SwiGLU itself ran on BF16.

## Verification

```text
CPU CTest                 158/158 pass
ASan/UBSan CTest          156/156 pass
HIP CTest                  61/61 pass
PyTorch operator oracle      4/4 pass
raw benchmark records       36/36 accuracy + zero payload transfer
```

The PyTorch oracle is Python `torch`, not LibTorch. It performs BF16 gate/up matmul,
BF16 SwiGLU, BF16 down matmul and FP32 boundary conversion using the same shapes and
values.

## Decision boundary

Keep the operator, real-shape fallback, tests, benchmark and records. Do not yet switch
Qwen or DeepSeek as a whole. The remaining model gate must load only one persistent
inference weight representation, compare official logits/tokens against PyTorch BF16,
and measure the entire train/generate matrix.

## Falsification

If model integration needs duplicate persistent weights, changes exact tokens beyond the
declared BF16 tolerance, or loses end-to-end throughput, keep this operator as a measured
primitive but reject that model policy. BF16 results stay on their separate figure and
never alter the FP32 score.
