# Experiment 216 — 优化以后，时间又去了哪里

Status: `measured; optimizer threshold track closed`

## 为什么必须重新 profile

Experiment 213 说 AdamW 占 Qwen/DeepSeek Kernel 时间的 16.85%/21.52%。Experiment 214/215
改变了 moment 字节数和提交方式，因此旧比例已经不能继续指导下一步。

本实验对同一二进制分别采集：

```text
load + 1 step
load + 3 steps
--------------
差值 / 2 = 一个稳定训练 step
```

四份 trace 的 Kernel call 差都非负。加载专属的 BF16→FP32 转置与初始化 fill 被排除，不会再
冒充训练热点。

## 新时间分布

| Category | Qwen | DeepSeek |
|---|---:|---:|
| hipBLASLt GEMM | 59.33% / 19.054 ms | 63.81% / 46.519 ms |
| AdamW | 12.82% / 4.118 ms | 17.61% / 12.836 ms |
| Other kernels | 5.11% | 3.73% |
| RMSNorm | 4.01% | 2.77% |
| Bias gradient | 4.06% | 2.41% |
| Cross entropy | 3.97% | 1.99% |
| FP32→BF16 cast | 3.06% | 2.66% |

![Post-hybrid training profile](../assets/post-hybrid-training-profile.svg)

## 与旧 profile 的闭环

| Model | Old Kernel/step | New Kernel/step | Total speedup | AdamW speedup |
|---|---:|---:|---:|---:|
| Qwen | 33.543 ms | 32.117 ms | 1.044× | 1.372× |
| DeepSeek | 77.126 ms | 72.906 ms | 1.058× | 1.293× |

新 Qwen AdamW 是73个大Tensor Kernel加1个小Tensor Kernel；DeepSeek是142+1。说明阈值策略
确实完成了目标，不应继续扩大阈值。GEMM 现在占绝对多数，单独完美删除其他任一小类别的上限
仍很低。

## 下一合同

下一代码节点必须改变训练 GEMM 的结构：例如相同输入的 grouped weight-gradient GEMM、能够跨
Q/K/V或gate/up提交的稳定计划，或针对真正 top shape 的可复现方案。它必须：

1. 完整对齐所有 weight/input gradient；
2. 至少命中 Qwen 和 DeepSeek 各一个 top GEMM family；
3. 两模型端到端均达到 `1.01×`；
4. 不增加超过 1% 峰值显存；
5. 失败时删除模型路由，保留算子证据。

原始证据在
[`benchmarks/results/2026-08-24-post-hybrid-training-profile/`](../../../benchmarks/results/2026-08-24-post-hybrid-training-profile/)。
