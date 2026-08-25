# Experiment 271 — overlap不是开关，而是context策略

Status: `context-selective keep; general default remains off`

固定Model-S、两rank、三步和25 MiB，只比较同步views/overlap views。T32/T128每个
context/policy各三个fresh进程、6个steady样本。

| Context | Sync total | Overlap total | Total ratio | Finish ratio | Peak delta |
|---:|---:|---:|---:|---:|---:|
| 32 | 8.015 ms | 8.019 ms | 0.9995× | 2.022× | 0 |
| 128 | 9.289 ms | 8.504 ms | 1.0923× | 2.235× | 0 |

![Ranked overlap context scale](../assets/ranked-overlap-context-scale.svg)

T32复现中性结果；T128稳定过1.01门。T128同步CV 6.80%、overlap 2.21%，因此保留原始分布。
即使删除整个最慢的process_run 1，T128同步/overlap仍为9.093/8.504ms，speedup `1.069×`。

T128 F/B added只有0.466ms，而finish少1.682ms；T32 added为1.178ms，抵消finish收益。两尺度
current/peak增量均0。T128 CPU Max/RMS为`0.003842/2.595e-6`，loss差`1.812e-5`；全部rank
exact、故障门通过。

结论不是“默认开启overlap”，而是“在当前Model-S/two-MI300X/25MiB轨道，T32关闭、T128
准入”。其他model、GPU、world size、bucket limit和context仍需独立证据。

证据：[`ranked overlap contexts`](../../../benchmarks/results/2026-08-25-ranked-overlap-contexts/)
