# Experiment 119 data

MI300X/gfx942 上执行 dense square GEMM，计算 achieved TFLOPS、官方峰值利用率和 5.3 TB/s
roofline 利用率。

```text
sizes        128, 256, 512, 1024
paths        readable FP32, hipBLASLt FP32/FP16/BF16/FP8 E4M3-FNUZ
warmup       5
repetitions  20
timing       HIP Event kernel time
FP8 output   BF16
sparsity     none
```

- `raw.jsonl`：20 条 executed dtype/size 记录；
- `summary.json`：每个 dtype 的最佳尺寸、TFLOPS、峰值和 roofline 利用率；
- `gpu2-preflight.jsonl`：正式运行前连续三次 0/0；
- `gates.json`：环境、精度、测试和结论边界。
