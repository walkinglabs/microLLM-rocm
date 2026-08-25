# Step 88 — Rank-local gradient buckets

Status: complete, correctness baseline kept

one-process-per-GPU worker当前对tiny的12个参数分别发起collective。下一步先实现rank-local同步
bucket reducer：按相同parameter order/byte limit pack到本rank稳定Tensor，单Tensor RCCL average，
再以view或unpack恢复optimizer gradient。

必须证明：bucket count与payload精确、两rank/CPU参数等价、collective 12→自然bucket数、ID/timeout/
peer failure不退化。先做tiny和Model-S one-step smoke，不做overlap；同步bucket稳定后才迁移ready
hook和Event。

实现新增rank-local同步bucket API和`per-parameter|bucket` worker控制。pack、RCCL average、unpack
全在rank通信Stream按序执行，bucket销毁前同步；统计bucket/parameter/elements/copy。tiny 4096B
自然为1 bucket/step，3step collective 36→3，728值rank exact、CPU最大差1.19e-7。

world1 API、双进程bucket smoke与原有peer failure通过；正式三次两策略轮换前不迁移overlap。

正式结果：collective/rank 36→3（12×），参数/CPU/故障门通过，wall只有1.0037×。启动成本淹没
tiny通信，保留同步baseline但不作性能结论；下一步Model-S one-step自然3 bucket。
