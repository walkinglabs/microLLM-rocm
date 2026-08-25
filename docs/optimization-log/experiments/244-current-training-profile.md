# Experiment 244 — 现在训练时间花在哪里

Status: `measured; hotspot order unchanged`

本轮使用刚加入的一键 runner，在当前二进制上重新采集四个 MI300X 进程。每个模型分别运行
`load + 1 step` 与 `load + 3 steps`，相减后除以 2。这样加载权重、创建 optimizer 和首次
初始化不会冒充稳定训练时间。

## 当前结果

| Model | Kernel/step | GEMM | AdamW | Calls/step |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 31.327 ms | 58.56% | 13.22% | 1,698 |
| DeepSeek-R1-Distill-Qwen-1.5B | 71.873 ms | 63.43% | 18.16% | 2,037 |

![Current training profile](../assets/current-training-profile.svg)

相对 Experiment 216 的同口径结果，Kernel 总时改善 1.0252×/1.0144×。GEMM 占比仍远高于
任何单个小类；AdamW 占比略有波动，但提交数仍是 74/143，阈值搜索没有重新打开的证据。

四个应用记录都改变了参数，optimizer D2H 为零；H2D 只有每步固定 descriptor metadata：
Qwen 13,888 bytes、DeepSeek 12,608 bytes。两个派生 profile 都没有负 Kernel-call delta。

## 下一合同

下一实现必须改变训练 GEMM 或 graph-wide 结构，并继续要求：

1. 完整梯度或状态正确性先于计时；
2. 同时命中两个模型的真实 top family；
3. 至少做同二进制端到端 A/B；
4. 单个 Kernel 变快但整步不变时拒绝默认路由。

证据：[`current training profile`](../../../benchmarks/results/2026-08-25-current-training-profile/)

