# Step 88 — Rank-local gradient buckets

Status: planned

one-process-per-GPU worker当前对tiny的12个参数分别发起collective。下一步先实现rank-local同步
bucket reducer：按相同parameter order/byte limit pack到本rank稳定Tensor，单Tensor RCCL average，
再以view或unpack恢复optimizer gradient。

必须证明：bucket count与payload精确、两rank/CPU参数等价、collective 12→自然bucket数、ID/timeout/
peer failure不退化。先做tiny和Model-S one-step smoke，不做overlap；同步bucket稳定后才迁移ready
hook和Event。
