# Experiment 133：同一reduction让冷启动快73倍、T512快21倍

## Suite A：weight准备

| 模型 | single-block | multi-block | 加速 | FP8 TPS | RMS |
|---|---:|---:|---:|---:|---:|
| Qwen | 500.95ms | 20.34ms | **24.63×** | 1883 | 0.6644 |
| DeepSeek | 2111.99ms | 28.82ms | **73.27×** | 1375 | 1.1112 |

## Suite B：T512 activation

| 模型 | single-block TPS | multi-block TPS | 加速 | /BF16 | RMS |
|---|---:|---:|---:|---:|---:|
| Qwen | 4874 | 75518 | **15.50×** | 0.818× | 0.2925 |
| DeepSeek | 2181 | 44975 | **20.62×** | 0.908× | 0.2491 |

![Multi-block amax](../assets/fp8-multiblock-amax.svg)

两套max/RMS与单block基线逐值相同，top token一致，证明这是纯性能优化。完整FP8精度门仍全部
失败，吞吐也仍未超过BF16；因此保留Kernel，不改变模型精度结论。

multi-block已经关闭“一块GPU却只用一个block扫描”的主要解释。剩余性能差距来自量化Kernel、
额外reduction/临时Tensor、FP8 GEMM效率和fallback；下一步必须profile正式T512时间线，不能继续
只改reduction。
