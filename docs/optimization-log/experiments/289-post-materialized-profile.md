# Experiment 289：新默认把时间搬到哪里了

Status: profile complete; exact-order finalize selected

## 为什么必须重测

Experiment 288让一条有边界的新Attention路径成为长上下文默认。旧profile测的是旧程序；继续拿
旧的61.57%当事实，就像换了发动机后仍拿旧油耗表决定下一步。

本实验固定DeepSeek、T2048、B2、BF16 KV和64个新token。一个进程做1次generation，另一个做
3次；两者都有相同加载和热身。用`(three-one)/2`得到一次稳定generation，并要求两个进程都
报告`auto-enabled`。

![Post-materialized profile](../../../benchmarks/results/2026-08-25-post-materialized-deepseek-t2048-profile/profile-delta.svg)

## 看见了什么

| 阶段 | 每代Kernel时间 | 占比 | calls |
|---|---:|---:|---:|
| cached Attention finalize | 349.17 ms | 42.00% | 1,792 |
| hipBLASLt GEMM | 272.79 ms | 32.81% | 12,861 |
| cached Attention scores | 64.81 ms | 7.80% | 1,792 |
| 其他全部阶段 | 144.54 ms | 17.39% | — |

Attention合计413.99ms/49.80%，仍是最大子系统，但现在可以清楚地区分score与finalize。旧路径
Attention为647.26ms；新路径合计快1.5635x。整个Kernel从1051.29ms降到831.31ms，历史同
workload比为1.2646x；应用generation为776.14ms，历史比为1.2774x。历史比较不是交错A/B，
所以只用于解释阶段移动，不替代Experiment 288的正式性能门。

## 一个重要反例

新路径每次调用需要score Tensor，不等于steady decode正在向GPU反复申请显存。差分中38,755次
逻辑申请全部命中cache，backend allocation增量为0。若此时先写workspace，只会优化不存在的
稳态分配热点。

## 决定

Step 106完成。下一节点只比较exact-order finalize的线程映射；score Kernel、模型路由和默认
策略保持不变。候选必须先通过完整context位级/误差门，再通过官方DeepSeek完整logits、token、
峰值显存和端到端门。若所有映射都失败，再把GEMM提升为下一架构级候选。

证据：[`post-materialized profile`](../../../benchmarks/results/2026-08-25-post-materialized-deepseek-t2048-profile/)
