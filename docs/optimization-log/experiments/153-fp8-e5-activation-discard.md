# Experiment 153：E5范围更大，但完整误差最多恶化3.43倍

本实验只改变activation格式：candidate使用E5M2-FNUZ，control使用E4M3-FNUZ；Linear权重
始终保持E4M3-FNUZ。两者来自同一revision、同一fresh binary、同一O-only权重scope与动态
Tensor amax。两套各36个worker，共比较24个FP8完整logits行。

| 模型/上下文 | Max E5/E4 | RMS E5/E4 | TPS变化 | 显存变化 |
|---|---:|---:|---:|---:|
| Qwen T8 | 1.582× | 1.588× | +2.16% | 0 |
| Qwen T512 | 1.508× | 2.123× | +0.06% | 0 |
| DeepSeek T8 | 2.058× | 2.348× | -0.18% | 0 |
| DeepSeek T512 | 2.848× | 3.428× | -0.30% | 0 |

![E5 activation format discard](../assets/fp8-e5-activation-discard.svg)

八项Max/RMS全部变差，完整precision仍是0/4。两项T512吞吐均在5%门内且resident/peak增量
为零，但性能和显存不能抵消明确的数值回归。动态amax已经把输入映射到可表示范围；E5新增
指数范围没有解决当前问题，少一位尾数反而在传播中放大误差。

因此model、CLI和通用matrix中的E5策略被删除。底层Tensor dtype、量化/反量化、独立左右
operand dtype的autograd API，以及MI300原生E5×E4 GEMM测试继续保留：它们证明硬件原语
存在，不能被写成当前Qwen/DeepSeek模型策略可用。

下一步不能继续只换全模型activation格式。应测量每个Linear输入在完整数据上的分布并研究
分层静态校准；候选仍必须回到同revision的native完整logits、速度和显存门。
