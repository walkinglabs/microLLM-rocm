# Step 78 — Bucket pack/unpack and temporary attribution

Status: planned

扩展BucketStats：bucket Tensor数、pack/unpack copy数、临时元素/字节；DataParallel metrics单列通信
阶段逻辑/backend分配delta。Model-S 25MiB/3bucket应得到可手算恒等式。只有确认pack/unpack和
temporary是热点，才设计persistent buffer或gradient-as-bucket view。

