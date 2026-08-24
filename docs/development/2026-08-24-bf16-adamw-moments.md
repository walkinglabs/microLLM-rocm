# 2026-08-24 — BF16 AdamW moment 节点

## 目标

Experiment 213 指出 AdamW 是 Qwen/DeepSeek 两步训练 Kernel 时间的 16.85%/21.52%。本节点
不再只减少 launch，而是把两个 moment 的实际存储从 FP32 改为 BF16。

## 实现边界

- 默认 FP32 不变；`AdamWConfig` 和两个训练 CLI 增加显式 BF16 策略；
- CPU/HIP 都按“先舍入 moment，再更新 parameter”执行；
- FP32 master、gradient 和 BF16 weight mirror 的语义不变；
- checkpoint 升级到 v2，保存策略和 canonical CPU FP32 state；v1 继续可读；
- 扩展 multi-tensor 原语支持 BF16，但失败的全量模型路由没有保留；
- 修正 `hf_train_step` 的 optimizer 计时边界，避免混入异步 backward 尾部。

## 证据

- CPU：2 步和 100 步手算轨迹；
- PyTorch：第 2/32 步 parameter、first/second moment、mirror；
- HIP：单/多 Tensor 完整状态、尾部 shape 和传输计数；
- checkpoint：BF16 恢复轨迹、策略拒绝和真实 v1 二进制兼容；
- 官方模型：B1/T512、1 warm-up、2 measured、每策略 5 个新进程。

正式结果为 Qwen/DeepSeek端到端 `1.0226×/1.0356×`，optimizer
`1.0687×/1.1964×`，峰值 `0.8329×/0.8084×`。状态为 partial keep；Qwen 的
`1.10×` optimizer stretch gate 没有通过。

最终发布门：CPU 324/324、ASan/UBSan 322/322、PyTorch 298/298、CPU/HIP
505/505（3 个条件跳过、HIP 标签 172/172）、RCCL 14/14（multi-GPU 11/11）。覆盖率为
80.1% lines、87.9% functions、60.6% branches；覆盖清单注册 87 个测试文件。

完整实验见
[Experiment 214](../optimization-log/experiments/214-bf16-adamw-moments-partial.md)，原始数据见
[`benchmarks/results/2026-08-24-bf16-adamw-moments/`](../../benchmarks/results/2026-08-24-bf16-adamw-moments/)。
