# Experiment 245 — 低精度 weight gradient 不能一刀切

Status: `operator admitted; gate/up only enters model gate`

当前 BF16 Linear 的 forward 已经低精度，但 weight gradient 仍使用 FP32。候选执行：

```text
cast+transpose(input) + cast(dY) + BF16 GEMM -> FP32 dW
```

计时包含两次 cast。每个真实 B1T512 shape 启动三个新进程。

| Model | Family | Median | Minimum | Decision |
|---|---|---:|---:|---|
| Qwen | query | 0.718× | 0.718× | reject |
| Qwen | KV | 0.821× | 0.813× | reject |
| Qwen | gate/up | 1.459× | 1.446× | model gate |
| DeepSeek | query | 0.976× | 0.965× | reject |
| DeepSeek | KV | 0.816× | 0.813× | reject |
| DeepSeek | gate/up | 1.890× | 1.823× | model gate |

![BF16 weight-gradient shapes](../assets/bf16-weight-gradient-shapes.svg)

18份完整输出都有限，BF16 CPU抽样最大误差不超过5.22e-8。与FP32基线的完整Max/RMS
也保留在原始记录中，因为这里确实改变了训练精度，不能写成bit-exact。

正式 `bf16_weight_gradient` API 已通过 CPU、HIP 与 PyTorch BF16数学对齐。Autograd只给
gate/up提供显式开关，默认关闭；query/KV四个反例禁止建立全局低精度梯度策略。

下一步是同二进制官方模型 A/B。算子加速不能直接成为训练吞吐结论。

证据：[`operator matrix`](../../../benchmarks/results/2026-08-25-bf16-weight-gradient-operator/)

