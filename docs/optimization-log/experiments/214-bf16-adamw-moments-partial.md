# Experiment 214 — 用 BF16 缩小 AdamW 的两本笔记

Status: `partial keep`

## 问题不是启动次数，而是实际读写字节

Experiment 211 把几百次 AdamW launch 合成一次，DeepSeek 仍只有 `1.0094×`。Experiment 213
又证明 AdamW 占真实两步 Kernel 时间的 16.85%/21.52%。这说明下一次实验必须减少参数状态的
真实流量。

普通 AdamW 为每个参数保存两个 FP32 moment，共 8 B/parameter。本实验把它们改为 BF16，
共 4 B/parameter；FP32 master、gradient 和可选 BF16 weight mirror 都不变。

## 正确性先于速度

更新顺序固定为：FP32 计算 recurrence，先舍入并保存 BF16 moment，再用舍入值更新 FP32
parameter。证据包括：

- CPU 两步逐项参考和 100 步变化梯度轨迹；
- HIP 4,099 尾部 shape，parameter、两个 moment、mirror 全状态对齐；
- PyTorch 第 2/32 步四类状态对齐；
- checkpoint v2 保存策略、导出 canonical FP32 moment、恢复后后续轨迹相同；
- 真实二进制 v1 checkpoint 按 FP32 策略继续加载；
- optimizer 测量区间没有 H2D/D2H Tensor payload。

## 测量边界也需要修复

第一版 `optimizer_ms` 在异步 backward 后立刻启动主机计时，因而混入未完成的 backward 尾部。
正式数据在 backward 完成后建立计时边界；端到端 `measured_ms` 仍覆盖整个训练 step。历史 pilot
保留，不能与修正后的纯 optimizer 数字混用。

## 两个反驳实验

1. BF16 single-tensor Kernel 对两个大 embedding shape 达到约 `1.24×–1.26×`，但旧的
   multi-tensor 方式只有一个 element/thread；
2. 将 multi-tensor 扩展到 BF16 并让每线程处理四个元素后，三进程 pilot 仍使 Qwen
   optimizer/吞吐退化，DeepSeek 也没有形成可保留的共同结果。

因此训练器保留逐 Tensor BF16 Kernel；BF16 multi-tensor 只作为经过测试的研究原语存在，
没有偷偷进入模型路由。

## 五进程正式结果

BF16 Linear + FP32 master，B1/T512，一次热身、两步测量；策略顺序交替，每个策略五个新进程。

| Model | FP32 moment | BF16 moment | End-to-end | Optimizer | Peak ratio | Final-loss relative diff |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,896.09 | 15,233.46 tok/s | 1.0226× | 1.0687× | 0.8329× | 0.00129% |
| DeepSeek Distill 1.5B | 6,192.87 | 6,413.17 tok/s | 1.0356× | 1.1964× | 0.8084× | 0.00326% |

两模型的 moment bytes 都精确减半，端到端、峰值显存和 1% loss 门通过。Qwen optimizer 未达到
预先设置的 `1.10×` stretch gate，所以状态是 `partial_keep`。默认策略仍为 FP32；BF16 通过
`--adamw-moment-precision bf16` 显式启用。

![BF16 AdamW moments](../assets/bf16-adamw-moments.svg)

## 下一步

Qwen 剩余差距主要来自许多小 Tensor 的固定 launch/转换成本。下一节点只能尝试“仅合并小
Tensor、保留大 Tensor 向量路径”的分层策略，并必须同时超过当前逐 Tensor正式结果；不能重新
启用本实验已经否决的全量 multi-tensor 路由。

原始数据和复现命令在
[`benchmarks/results/2026-08-24-bf16-adamw-moments/`](../../../benchmarks/results/2026-08-24-bf16-adamw-moments/)。
