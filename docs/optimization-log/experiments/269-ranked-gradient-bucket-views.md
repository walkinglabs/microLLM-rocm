# Experiment 269 — 去掉unpack以后，常驻显存回到基线

Status: `kept explicit; overlap prerequisite admitted`

固定Model-S、两rank、三步、25 MiB。四策略各三个fresh进程组；每策略6个steady样本。

| Policy | Reducer | Total | Current | Peak | Unpack |
|---|---:|---:|---:|---:|---:|
| per-parameter | 2.619 ms | 8.693 ms | 249.38 MB | 262.58 MB | 0 |
| transient | 4.692 ms | 10.497 ms | 249.38 MB | 314.90 MB | 57 |
| persistent-copy | 2.981 ms | 8.292 ms | 311.72 MB | 387.27 MB | 57 |
| bucket-views | 2.662 ms | 8.242 ms | 249.38 MB | 324.93 MB | 0 |

![Ranked gradient views](../assets/ranked-gradient-bucket-views.svg)

views相对persistent-copy Reducer/total为`1.120×/1.006×`，相对transient为
`1.763×/1.274×`。相对逐参数Reducer为`0.984×`，完整step为`1.055×`。

plan容量从124.69MB减半到62.34MB；unpack 57→0，warmup后backend allocation保持0。final
current回到逐参数水平；peak仍比逐参数多62.34MB，但比persistent-copy少62.34MB。

24个rank进程的完整参数、CPU、loss与故障门通过。views显式保留、不默认；它满足真实
one-process-per-GPU gradient-ready overlap的Storage前置条件。下一节点只改变communication
enqueue时机，保持views、容量与显存合同不变。

证据：[`ranked gradient views`](../../../benchmarks/results/2026-08-25-ranked-gradient-views/)
