# Step 80 — Persistent bucket and unpacked-gradient storage

Status: planned

为固定parameter order/device/world/bucket limit构建move-only plan，长期持有rank bucket与114个
unpacked gradient Tensor。第一步建plan并做完整门，后续通信allocation/backend目标120→0，
pack/unpack copy暂保留。必须检查gradient set/zero生命周期、地址稳定、loss/参数、peak/current
bytes和Model-S端到端；未过门则删除模型route。

