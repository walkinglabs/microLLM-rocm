# Qwen3 BF16 optimizer 状态压缩

日期：2026-08-27

两条路使用完全相同的官方Qwen3 BF16 forward和梯度，只改变AdamW记忆的存储格式。FP32 moments占4.768GB，BF16 moments占2.384GB，精确减半。

三步附加误差为：loss `0.002605`，参数Max/RMS `3.470e-5/1.197e-6`，canonical moments `0.04736/3.335e-5`，六门全过。Auto混合调度把169个不超过1,048,576元素的Tensor放进multi-tensor路径，共58,785,792元素。

边界必须写清：上游BF16相对PyTorch已经失败。本节点只说明在该实验路径里压缩optimizer state的额外误差受控且节省2.384GB，不能称为BF16训练对齐。
