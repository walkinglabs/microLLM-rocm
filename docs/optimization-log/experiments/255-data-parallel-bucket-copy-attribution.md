# Experiment 255 — 三个bucket为什么每步有126次显存申请

Status: `persistent reducer design admitted`

Model-S 25MiB/3bucket，干净commit，step2 steady：

```text
6 bucket + 6 average + 114 unpacked = 126 Tensor/backend allocations
114 pack + 114 unpack = 228 D2D copies
15,586,176参数 × 2 ranks × 3份 = 374,068,224 bytes
```

![Data parallel bucket copy attribution](../assets/data-parallel-bucket-copy-attribution.svg)

通信allocation/backend均为126，cache reuse为0，allocated bytes与temporary bytes逐字节相同。
RCCL non-default stream会关闭exact-size pool，因此这些是每步真实backend申请，不是逻辑计数。

steady communication 7.26ms，占22.47ms total的32.31%。persistent reducer设计准入，但必须
覆盖bucket、average和114个unpacked gradient；只缓存6个bucket不能过门。

第一小步先让average原地缩放，保持bucket地址稳定并删除6个Tensor。随后再构建persistent
bucket/unpacked plan。

证据：[`copy attribution`](../../../benchmarks/results/2026-08-25-data-parallel-bucket-copy-attribution/)

