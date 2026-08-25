# Experiment 267 — cold说快，steady为什么反而慢48%

Status: `transient bucket rejected for steady performance`

固定Model-S、两rank、`B1×T32/rank`、三步和25 MiB；两策略各三个fresh进程组。第1步完整
保留为cold，步骤2–3共形成每策略6个steady样本。

| Policy | Cold Reducer | Steady Reducer | Steady CV | Steady total |
|---|---:|---:|---:|---:|
| per-parameter | 53.93 ms | 2.837 ms | 27.74% | 8.864 ms |
| transient bucket | 40.82 ms | 4.205 ms | 2.72% | 10.396 ms |

![Ranked steady reducer](../assets/ranked-steady-reducer-discard.svg)

只看cold会得到bucket快1.321×；只看包含cold的进程总量也会得到错误方向。steady中，bucket
Reducer speedup为`0.6747×`，也就是用时多48.2%；完整step speedup为`0.8527×`，用时多
17.3%。bucket steady CV只有2.72%，反例稳定。

归因也逐项闭合：每个steady bucket step有60次logical/backend allocation、60次deallocation、
124,689,408 bytes分配，以及57 pack + 57 unpack；per-parameter这些计数均为0。collective虽从
57降到3，但临时Storage和114次copy超过收益。

三步后15,586,176个参数值跨rank exact，CPU Max/RMS为`0.0062715/3.701e-6`，loss差
`1.967e-5`，故障门不退化。transient bucket保留为正确性实现，但拒绝作为steady性能路线。

下一反驳实验只改变Storage生命周期：建立rank-local persistent plan，要求warmup后backend
allocation为0，再测Reducer、完整step、live/peak显存。ready overlap仍不提前接入。

证据：[`ranked steady reducer`](../../../benchmarks/results/2026-08-25-ranked-model-s-steady-reducer/)
