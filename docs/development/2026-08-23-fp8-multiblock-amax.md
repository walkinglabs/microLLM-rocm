# FP8 multi-block Tensor amax

旧device amax只用一个256-thread block扫描完整Tensor。新实现最多使用1024个blocks，每个block
写一个partial maximum，再由一个block完成最终scale。量化Kernel在同一Stream中等待scale，
不增加host同步。

GPU workspace是短生命周期FP32 partial Tensor；默认Stream allocator的顺序复用合同保证它在
Kernel完成前不会被错误覆盖。

新增262,144元素反例，把唯一最大值`-123`放在最后partition。测试证明scale为123/240，动态
量化阶段0 H2D/0 D2H，解量化恢复末尾值。原有tiny、prepared model、device weight amax和
fallback门全部继续通过。

完整Release/MI300回归348/348通过，2个条件跳过，包含fresh CLI binary contract。性能结论仍需
分别重跑后，Qwen/Deep weight准备提升24.6×/73.3×，T512 activation提升15.5×/20.6×，且
max/RMS逐值不变。详见[Experiment 133](../optimization-log/experiments/133-fp8-multiblock-amax.md)。
