# Experiment 250 — 当前训练局部策略到边界了吗

Status: `local default-policy search saturated`

当前B1T512 profile中，GEMM占58.56%/63.43%，AdamW占13.22%/18.16%。剩余cast即使
免费删除，Kernel-only上限也只有1.0332×/1.0277×；最大的其他小类别完美删除上限也只有
1.0507×/1.0388×。

![Training local saturation](../assets/training-local-saturation.svg)

最近六条相邻路线已经分别由能力、完整输出、端到端、长轨迹或workspace门关闭：

1. grouped weight gradient；
2. packed weight gradient；
3. exact weight-gradient solution；
4. optimizer-only Graph model route；
5. BF16 gate/up weight-gradient trajectory；
6. BF16 weight-gradient workspace。

结论不是“训练无法优化”，而是现有默认路径上的局部旋钮不值得继续拨。下一任务必须改变
custom kernel/graph-wide架构尺度，或转向production data-parallel reducer。没有新backend、
硬件矩阵或反驳合同，不重开已关闭track。

证据：[`saturation package`](../../../benchmarks/results/2026-08-25-training-local-saturation/)

