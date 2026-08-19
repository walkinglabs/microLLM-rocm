# 2026-08-19 — MI300X FP8 training and inference operators

## Contract

MI300X/gfx942 uses FNUZ FP8. The public scaled Tensor convention is:

```text
quantized = round(source / scale)
dequantized = quantized * scale
```

`ScaledTensor` keeps one-byte FP8 values, a same-device FP32 scale Tensor for
hipBLASLt, and the host scale value used when launching conversion kernels.

## Implemented

- E4M3-FNUZ and E5M2-FNUZ CPU reference conversion;
- device-native FP32/FP16/BF16 to FP8 quantize Kernel;
- device-native FP8 to FP32/FP16/BF16 dequantize Kernel;
- hipBLASLt FP8 × FP8 scaled GEMM with FP32/FP16/BF16 output;
- eager-autograd FP8 forward with FP32 master parameters and FP32 backward gradients;
- Transformer `LinearPrecision` policy covering Q/K/V/O, FFN, output head, full
  forward/loss/backward, and one-token KV-cache decode;
- invalid format/scale gates and zero Tensor-payload host-transfer checks;
- unified precision benchmark with Event median/p95, accuracy gate, and speedup ratios.

The current training rule uses a quantized forward and a full-precision straight-through
backward. Dynamic amax/history scaling remains later work; explicit per-policy activation
and weight scales make the current behavior reproducible.

## Measured MI300X result

Shape `512 × 512 × 512`, five warm-ups and twenty measured repetitions:

| candidate | median ms | p95 ms | max abs error | vs readable FP32 | vs hipBLASLt FP32 |
|---|---:|---:|---:|---:|---:|
| readable FP32 | 0.227497 | 0.261798 | 6.56e-7 | 1.00x | 0.20x |
| hipBLASLt FP32 | 0.045343 | 0.070629 | 1.43e-6 | 5.02x | 1.00x |
| hipBLASLt FP16 | 0.041555 | 0.077623 | 5.37e-4 | 5.47x | 1.09x |
| hipBLASLt BF16 | 0.049432 | 0.264672 | 6.54e-3 | 4.60x | 0.92x |
| hipBLASLt FP8 E4M3-FNUZ | 0.040251 | 0.124820 | 5.55e-2 | 5.65x | 1.13x |

The result is deliberately shape-specific. FP8 wins the median here but has larger
error and tail latency; a shape matrix is required before making a general claim.
