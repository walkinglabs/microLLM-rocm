# Experiment 011 — hipBLASLt projection bias epilogue

Status: `discard`

## Observed bottleneck

Qwen decode trace launches 792 standalone `add_bias_kernel` calls, about 5.6% Kernel
time. Attention Q/K/V Linear performs GEMM, allocates another Tensor, then adds bias.

## Hypothesis

hipBLASLt's bias epilogue can add the output-width vector inside projection GEMM,
removing one Tensor and one launch. Official generation should improve without changing
training autograd or FP8.

## Scope

- FP32 2D inference Linear with bias;
- hipBLASLt epilogue when optimized GEMM is selected;
- CPU/readable fallback remains `matmul → add_bias`;
- training autograd, FP8, no-bias Linear, allocator and model weights unchanged.

## Required gates

- [x] hand/PyTorch forward and invalid bias shape
- [x] forced large-shape CPU/hipBLASLt parity
- [x] zero payload host transfer
- [x] exact Qwen/DeepSeek tokens
- [x] three-process inference medians

## Candidate

- added a FP32 `matmul_bias` operator with CPU/readable fallback;
- row-major output is submitted as column-major `C^T`, whose row count is the user
  output width, so hipBLASLt's bias-vector rule matches the model bias;
- inference Linear with bias selected the epilogue; training autograd stayed unchanged;
- focused CPU, external oracle and 128×128 MI300X parity passed.

## Three-process medians

| Model | Fused-Attention baseline median | Bias candidate samples | Candidate median | Change |
|---|---:|---|---:|---:|
| Qwen generation | 142.25 | 130.93 / 131.21 / 131.23 | 131.21 | -7.8% |
| DeepSeek generation | 53.04 | 55.56 / 54.13 / 53.59 | 54.13 | +2.1% |

Training is unchanged and reuses Experiment 009 baseline medians. The resulting
candidate score is `1.725932`, below the `1.752183` running best.

## Allocation result and interpretation

The intended structural change did happen:

```text
Qwen generation logical allocations      11,145 → 9,345
DeepSeek generation logical allocations  48,545 → 40,565
```

Nevertheless Qwen is slower. The likely explanation is that the fused epilogue selects
or executes a less favorable GEMM path for Qwen's projection shapes. Fewer Kernels is
not sufficient end-to-end evidence.

## Cleanup

Candidate code, public API, tests and oracle rows were removed. Framework source is
identical to Experiment 009 before this report is committed.

## Results

Falsified for the selected matrix: allocation/launch reduction did not imply faster
Qwen projection.

## Decision

`discard`. No bias-epilogue path remains.
