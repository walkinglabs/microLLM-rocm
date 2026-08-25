# Experiment 266 — collective少19倍，为什么仍不能宣布通信加速

Status: `kept as measured correctness baseline`

固定Model-S `B1×T32/rank`、一步、两rank和25 MiB。per-parameter与bucket各运行三个fresh
进程组，顺序交错；每次都用CPU `B2×T32`检查全部57个Tensor、15,586,176个参数值。

| Policy | Collective/rank | Reducer median | Reducer range | Training median | Group wall |
|---|---:|---:|---:|---:|---:|
| per-parameter | 57 | 54.51 ms | 32.06–56.19 ms | 5657.56 ms | 9510.10 ms |
| 25 MiB bucket | 3 | 32.48 ms | 19.55–158.52 ms | 5648.32 ms | 9487.82 ms |

![Ranked Model-S buckets](../assets/ranked-model-s-buckets.svg)

collective减少19×，Reducer中位数表面改善1.678×，但bucket三次的CV为89.3%，一次冷启动达到
158.52ms。完整训练与组wall只有1.0016×/1.0023×。因此不能从这个one-step矩阵宣布steady
通信加速，更不能直接准入persistent或overlap。

正确性门全部通过：rank/rank Max/RMS为0；rank/CPU Max `0.0062738`、RMS `3.483e-6`；
rank loss均值与CPU global-batch最大差`9.555e-7`。故障返回`[1,-15]`，等待peer被终止。

下一步在同一fresh进程内跑多步，逐step记录Reducer；将第一次RCCL冷启动与后续steady step分开。
只有steady分布稳定后，才测persistent rank bucket的allocation/copy/显存代价。

证据：[`ranked Model-S buckets`](../../../benchmarks/results/2026-08-25-ranked-model-s-buckets/)
