# Experiment 308：B2先漂P×V，B4/B8先漂QK

Status: numerical source split by descriptor

## 为什么必须看完整值

只看前200,000个QK score时，B1/B2完全相同。但一个完整B1 score有50,331,648个值，前缀只能说明
最前面没错。我们用filtered binary trace比较所有值，并额外只统计causal可见下三角，避免未来遮罩区
的差异冒充真正来源。

固定DeepSeek T2048、Q=296100、K/V=292135、B1/2/4/8、两个fresh process。每个非B1 case还比较
同一process内两条相同输入行。

## 结果

| Batch | causal-visible scores | probabilities | P×V | 首差 |
|---:|---:|---:|---:|---|
| B2 | exact | exact | Max `9.775e-6` | P×V |
| B4 | Max `0.03125` | Max `0.0052773` | Max `0.00033218` | QK |
| B8 | Max `0.03125` | Max `0.0052773` | Max `0.00033218` | QK |

B4/B8 raw score的第一个差异在索引2，确实位于未来遮罩区；但causal可见区域的第一个差异在索引2048，
所以“只有被mask的值不同”被推翻。probability首差在4097。B2的完整25,178,112个可见score和同量
probability仍位级相同，P×V在索引12,288才第一次不同。

所有B2/B4/B8内部两行在三阶段都位级相同，两次fresh process指标完全重复。临时二进制保留数为0。

![Attention core matrix](../../../benchmarks/results/2026-08-26-prefill-attention-core-matrix/attention-core.svg)

## 决定

不存在一个“只修P×V就覆盖全部batch”的单一路线。下一节点分两个descriptor做共同solution能力门：

1. QK：B1/B4/B8，M=N=2048、K=128、batch count随请求batch变化；
2. P×V：B1/B2，M=2048、N=128、K=2048。

只有完整输出、跨batch row-invariance和性能同时通过的候选才进入模型反驳；softmax不是任何batch的
独立首因，暂不修改。默认不变。

证据：[`result directory`](../../../benchmarks/results/2026-08-26-prefill-attention-core-matrix/)
