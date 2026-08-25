# Experiment 270 — wait快2.18倍，完整step为什么只快0.52%

Status: `overlap performance rejected for Model-S T32; route explicit`

固定Model-S、两rank、三步、25 MiB。五策略各三个fresh进程组；同步views和overlap views各有
6个steady样本。

| Policy | F/B or enqueue | Finish wait | Total | Current | Peak |
|---|---:|---:|---:|---:|---:|
| synchronous views | 5.257 ms | 3.080 ms | 8.195 ms | 249.38 MB | 324.93 MB |
| overlap views | 6.456 ms | 1.413 ms | 8.152 ms | 249.38 MB | 324.93 MB |

![Ranked gradient overlap](../assets/ranked-gradient-overlap-discard.svg)

finish wait改善2.180×，少1.667ms；但hook、Event、pack和RCCL enqueue让backward区间增加
1.199ms。完整step只有`1.0052×`，低于预设1.01门。同步/overlap total CV为4.10%/3.47%，
六样本分布重叠。

steps2–3每rank都按固定顺序overlap 3 buckets，later allocation为0，current/peak与同步views
完全相同。30个rank进程的完整参数、CPU、loss与故障门通过。

所以不能把“等待快2倍”写成“训练快2倍”。overlap实现保留为显式教学/研究入口，不默认；
Model-S T32 ranked reducer局部搜索关闭。下一次overlap实验必须改变context/模型/拓扑，建立新
track，不再微调同一T32路径。

证据：[`ranked gradient overlap`](../../../benchmarks/results/2026-08-25-ranked-gradient-overlap/)
