# Step 79 — In-place bucket averaging

Status: planned

新增CPU/HIP/PyTorch对齐的 `scale_in_place_`，Communicator在all-reduce sum后原地乘1/world，
不替换bucket Storage。门：完整RCCL16+、bucket地址保持、平均值精确、Model-S每步average Tensor
6→0、backend allocations 126→120，loss/参数不变。它也是persistent plan的地址稳定前提。

