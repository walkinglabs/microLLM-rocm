# 2026-08-19 — MI300X precision capability and matrix acceleration

## Evidence levels

The repository now separates four claims:

1. `storage`: Tensor really uses the declared number of bytes;
2. `kernel`: a device Kernel executes and passes a numerical reference;
3. `matrix`: a dedicated hipBLASLt/Matrix Core path accepts the dtype;
4. `faster`: repeated HIP Event measurements beat the readable baseline.

`tests/hardware/precision_capability_test.cpp` is the dedicated hardware-format gate.
It records GPU architecture, runtime version, hipBLASLt availability, FP8 variant,
MXFP4/MXFP6, and INT4 Matrix status. Unsupported architectures skip rather than inherit
an assumption from another GPU generation.

## MI300X result

```text
GPU                         AMD Instinct MI300X VF
architecture                gfx942 / CDNA3
native FP16/BF16            yes
native FP8                  yes, FNUZ variant
native INT8 Matrix          yes
native MXFP8/MXFP6/MXFP4    no
documented INT4 Matrix      no
packed INT4 software path   planned
```

FP16/BF16 add, multiply, scale, SiLU, SwiGLU, and readable GEMM execute without host
transfers. hipBLASLt FP16/BF16 GEMM also executes and matches the rounded CPU reference.
The Python PyTorch oracle independently covers 22 FP16/BF16 forward cases across add,
multiply, scale, GEMM, Embedding, Softmax, RMSNorm, SiLU, SwiGLU, RoPE, and
cross-entropy; the registered parity test passes.

## Measured 512 cubed GEMM

Three warm-ups and ten repetitions; times are HIP Event kernel means:

| dtype | readable ms | hipBLASLt ms | measured speedup |
|---|---:|---:|---:|
| FP32 | 0.280200 | 0.062715 | 4.47x |
| FP16 | 0.185413 | 0.048378 | 3.83x |
| BF16 | 0.306531 | 0.054760 | 5.60x |

Raw records are in
`benchmarks/results/2026-08-19-mi300x-precision-matmul/results.jsonl`. This is one fixed
shape, not a universal performance claim; shape matrices and FP8 are separate work.
