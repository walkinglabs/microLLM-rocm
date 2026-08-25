# Step 89 — Ranked Model-S natural buckets

Status: implemented, formal clean-revision measurement pending

tiny的12个参数只能形成1个自然bucket，且组wall几乎全部是进程启动。当前节点固定Model-S
`B1×T32`、一步、两rank和25 MiB，比较：

```text
57 parameter all-reduces
vs
3 natural bucket all-reduces
```

完整15,586,176个参数值通过临时safetensors比较：rank/rank要求Max/RMS均为0；rank/CPU
global-batch同时要求Max不超过`1e-2`、RMS不超过`1e-5`；两rank loss均值与global-batch
loss差不超过`1e-4`。权重比较后删除，不进入Git。

worker新增落盘之外的forward/backward、Reducer、optimizer和完整训练计时。正式结果同时报告
collective数、较慢rank训练时间、较慢rank Reducer时间与组wall；组wall包含启动，只能作为补充。

基础设施冒烟得到3个自然bucket、rank exact、CPU Max `0.0062738`/RMS `3.483e-6`，且
`DistributedRank.*` 5/5。下一步必须从已push的干净revision运行两策略；一次运行只能证明
可执行和数量级，不能支持稳定speedup结论。
