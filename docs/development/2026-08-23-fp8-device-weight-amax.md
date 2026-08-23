# FP8 device-only weight amax

旧`tensor-amax`在模型准备时把每个GPU权重带回CPU，Qwen/DeepSeek扫描1.43/6.17GB并耗时约
2.8/12.2秒。新`device-tensor-amax`复用host-optional `ScaledTensor`：amax、scale和量化权重均
留在GPU，fallback也用device scale反量化。

新增报告区分：

- `weight_bytes_scanned`：host扫描字节；
- `device_weight_bytes_scanned`：device扫描字节；
- `device_amax_tensors`：GPU amax Tensor数；
- `host_scale_summary_available`：是否能在不D2H时报告min/max。

HIP tiny模型证明准备阶段0 D2H、8/8权重device amax、扫描字节等于释放FP32字节；prepared热路径
0 H2D/0 D2H，lazy/prepared输出一致。完整回归346/346通过，2个条件跳过；sanitizer定向门通过。

当前amax仍是单block正确性Kernel，官方冷启动时间和完整logits尚未比较，不能提前声称更快。
