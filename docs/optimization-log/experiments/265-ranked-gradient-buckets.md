# Experiment 265 — collective少12倍，tiny为什么几乎没变快

Status: `kept as correctness baseline`

per-parameter与4KiB one-bucket策略各三个fresh双进程run，每rank 3step，继续逐项对照CPU并重跑
peer failure。

| Policy | Collective/rank | Rank-group median | Rank diff | CPU diff |
|---|---:|---:|---:|---:|
| per-parameter | 36 | 5287.78 ms | 0 | 1.19e-7 |
| bucket | 3 | 5268.46 ms | 0 | 1.19e-7 |

![Ranked gradient buckets](../assets/ranked-gradient-buckets.svg)

collective减少12×，wall只1.0037×。tiny三步完全被进程、ROCm和RCCL约5.27s启动成本淹没，不能
写性能提升。同步bucket作为正确性baseline保留，故障仍为[1,-15]且不挂死。

下一步用Model-S one-step构成自然多bucket workload，先看collective/payload/参数等价，再考虑
persistent rank bucket或ready overlap。

证据：[`ranked buckets`](../../../benchmarks/results/2026-08-25-ranked-gradient-buckets/)
