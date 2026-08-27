# Qwen3 全量 AdamW 状态对齐

日期：2026-08-27

一次参数更新不仅改变权重，还会留下“过去梯度的记忆”。AdamW 的 first moment 像平滑后的方向，second moment 像梯度大小的记录。只比较参数而不比较这两份状态，恢复训练时仍可能走向不同轨迹。

本节点直接复用 `AdamW::state()` 的 checkpoint canonical FP32 snapshot。310个独立参数各有两份状态，因此比较620个Tensor、1,192,099,840个值；step必须双方都为1。PyTorch的`exp_avg/exp_avg_sq`按同一个参数映射和Linear转置转换。

FP32 Max/RMS为`5.743e-5/3.552e-8`，固定门全过。BF16 forward仍使用FP32 moments，但Max/RMS为`0.03641/2.879e-5`，Max超过`0.01`，最坏是tied embedding first moment。这证明差异来自进入AdamW的BF16梯度，不是BF16 moment压缩。

小Qwen2 fixture还用两步训练验证28个状态Tensor、1,360元素和step=2。每种官方精度的两个临时文件约9.54GB，比较后删除；导出不计入性能。

边界：BF16 moment存储策略和多步状态轨迹仍需单独验证。
