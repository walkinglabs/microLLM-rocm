# 2026-08-25 — bucket通信里的126个临时Tensor

`BucketStats` 现在记录bucket、average、unpacked Tensor数量，pack/unpack copy数与临时字节。
DataParallel metrics同时记录通信阶段allocation/backend/cache-reuse/allocated-byte delta。

首个Model-S 25MiB smoke暴露并修复了计数问题：Runtime HIP allocation counters是进程级，
按device查询后相加会双计数；现在只采一次。

修正后的3-bucket恒等式：

```text
6 bucket + 6 average + 114 unpacked = 126 Tensor
114 pack + 114 unpack = 228 D2D copy
93,517,056 float = 374,068,224 bytes
communication allocation/backend = 126 / 126
```

非默认RCCL stream关闭了exact-size pool，所以cache reuse为0。这为persistent reducer提供了
真实backend分配证据，而不是只看逻辑对象数。

