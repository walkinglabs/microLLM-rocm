# Experiment 268 — 分配归零以后，速度与显存怎样交换

Status: `kept explicit; not default`

固定Model-S、两rank、`B1×T32/rank`、三步和25 MiB。逐参数、transient bucket、persistent
bucket各三个fresh进程组；步骤2–3形成每策略6个steady样本。

| Policy | Steady Reducer | Steady total | Backend alloc | Current | Peak |
|---|---:|---:|---:|---:|---:|
| per-parameter | 2.692 ms | 8.712 ms | 0 | 249.38 MB | 262.58 MB |
| transient | 4.440 ms | 10.311 ms | 60 | 249.38 MB | 314.90 MB |
| persistent | 2.886 ms | 8.251 ms | 0 | 311.72 MB | 387.27 MB |

![Ranked persistent buckets](../assets/ranked-persistent-buckets.svg)

persistent相对transient的Reducer/完整step为`1.539×/1.250×`，warmup后backend allocation
`60→0`。相对逐参数，完整step达到`1.056×`，但Reducer仍只有`0.933×`，用时多7.2%。

代价不能隐藏：plan容量124,689,408 bytes/rank；current比两控制多62,344,704 bytes，peak比
逐参数多124,689,408 bytes、比transient多72,376,320 bytes。每步57 pack + 57 unpack仍在。

三步完整参数、CPU、loss和故障门通过。persistent copy作为显式研究策略保留，但不默认启用。
下一实验让57个参数gradient直接成为bucket view，目标是删除独立unpacked Storage与57次unpack，
把current恢复到逐参数水平，并重新检查peak与steady Reducer。

证据：[`ranked persistent buckets`](../../../benchmarks/results/2026-08-25-ranked-persistent-buckets/)
