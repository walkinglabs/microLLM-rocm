# Experiment 321 — down的exact方案最差慢了近一半

## 结论

真实down descriptor有15个共同候选，只有296100 block exact。它在M2048/4096/8192/16384上的
speedup为`0.506/0.758/0.686/0.863×`。性能远低于0.95门，因此直接拒绝，不进入模型测试。

## 为什么这里可以停止

这不是一个接近门槛的候选。B1几乎慢一半，几何平均只有0.690×。继续增加down模型scope只会留下
已经知道不合格的路径。通用operator能力和失败证据保留，用户route不增加。

至此，长上下文跨batch数值加法顺序已经从Q/K/V追到down，每一步都有真实descriptor和反例。下一步
必须回到clean upstream重新profile，而不是继续追逐“位级一致”本身。

原始结果见
[`benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance`](../../../benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance/README.md)。
