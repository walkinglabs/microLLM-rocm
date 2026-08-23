# Experiment 120 data

大尺寸 dense square GEMM，显式使用 `reference=fp32`，避免 CPU 2048/4096 reference 主导运行。

```text
sizes        2048, 4096
paths        readable FP32, hipBLASLt FP32/FP16/BF16/FP8 E4M3-FNUZ
warmup       5
repetitions  20
timing       HIP Event
reference    hipBLASLt FP32 output
```

FP32 reference 只用于大尺寸低精度相对误差；独立 FP32 正确性仍由 Experiment119 的小尺寸 CPU
reference 和现有算子门提供。
