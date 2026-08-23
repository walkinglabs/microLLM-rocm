# Experiment 147：八项改善七项，只差Qwen长上下文RMS

Attention-only给每层Q/K/V/O权重逐列scale，FFN与LM head继续device Tensor-amax。候选和control
使用同一个`13dad27` binary，各跑36个worker。

| 模型/上下文 | Max变化 | RMS变化 | T512 TPS变化 |
|---|---:|---:|---:|
| Qwen T8 | -9.31% | -10.80% | 不作正式门 |
| Qwen T512 | -10.20% | **+8.91%** | -4.26% |
| Deep T8 | -8.70% | -7.75% | 不作正式门 |
| Deep T512 | -16.26% | -14.32% | -4.42% |

![Attention-only FP8 weights](../assets/fp8-attention-only.svg)

两模型T512性能都在5%门内，额外常驻只有Qwen 196,224B、Deep 400,960B。每worker的post次数
为96/112个Attention Linear×4次forward，即384/448；hot column quantize为0。

但预设keep要求八个Max/RMS都不变差。Qwen T512 RMS从control增加8.91%，所以keep=false。
完整FP8门仍0/4，不能因7/8改善就写成“几乎可用”。

## 下一步为什么先拆O projection

Q/K/V直接改变长上下文Attention分数和value读取，O projection位于Attention结果之后。当前唯一
失败恰好是长上下文分布误差，因此下一次只给O逐列scale，Q/K/V恢复scalar；这能检验非线性
Attention内部是否造成Qwen T512回归。Attention-only scope暂时保留为实验地基，不设默认。
