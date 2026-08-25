# Experiment 280：57次copy消失了，为什么不能继续保留

Status: discard for performance; research primitive retained

## 假设和双门

把57次pack copy和3次bucket scale融合成3次gather-scale，应该同时满足：

1. 相对同步`bucket-views`至少1.01x；
2. 改善Step 102已保留的1.0661x和8.687ms候选，而不只是重新赢一次较慢对照。

第二个门阻止“代码更复杂，但系统没有更快”的候选进入保留路线。

## 固定实验

- 两张MI300X VF、Model-S、T128、rank rows `[1,2]`；
- 3个25 MiB bucket；
- 同步与`gather-weighted-overlap`交替3轮；
- 每轮3步，丢弃第1步，每策略6个steady sample；
- CPU、rank、策略完整参数、loss、显存、失败传播门与Step 102相同；
- 候选必须每步0 pack、3 gather、3 descriptor copy、1,368 descriptor bytes。

## 结果

![Ranked gather-scale discard](../assets/ranked-gather-scale-discard.svg)

计数目标全部实现：57 pack→0，独立bucket scale→0，gather-scale为3；later backend allocation
仍为0。finish从2.506ms降到1.333ms，快1.880x。steady step从8.901ms降到8.778ms，只有
1.0140x。

这勉强高于同步1.01门，却比Step 102的8.687ms慢0.090ms，速度比只有0.9897x；同时增加
1,368 bytes current/peak和每步1,368-byte描述传输。第二个门失败，所以不能保留为更优路线。

逐轮速度比为0.9898x、1.0387x、1.0310x；一轮回退。leave-one为1.0242x、1.0078x、
1.0200x，其中一次低于1.01，敏感性也不支持升级。

正确性全部通过：三轮策略最终15,586,176个参数逐项Max/RMS 0/0，rank 0/0，CPU
0.004938/3.218e-6，loss差1.72e-5。临时权重删除，peer failure有界。

原始证据：[`ranked gather-scale`](../../../benchmarks/results/2026-08-25-ranked-gather-scale/)

## 决定

拒绝gather-scale性能路线，保留Kernel和显式policy作为可继续研究的独立原语；默认与当前最佳
仍是Step 102的显式`bucket-weighted-overlap`。当前ranked reducer局部优化线到此停止。下一步
必须重新profile端到端时间线，选择新的主导瓶颈，而不是继续在1ms左右的finish内堆复杂度。
