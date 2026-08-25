# Step 80 — Persistent bucket and unpacked-gradient storage

Status: implemented, Model-S A/B pending

为固定parameter order/device/world/bucket limit构建move-only plan，长期持有rank bucket与114个
unpacked gradient Tensor。第一步建plan并做完整门，后续通信allocation/backend目标120→0，
pack/unpack copy暂保留。必须检查gradient set/zero生命周期、地址稳定、loss/参数、peak/current
bytes和Model-S端到端；未过门则删除模型route。

实现边界：默认关闭；plan绑定communicator devices、parameter identity/order、shape与bucket
limit，契约变化必须先clear。首步建立6个bucket和114个unpacked Tensor，后续step复用地址，
通信阶段allocation/backend目标为0。`temporary_bytes`只描述逐步临时Storage，持久容量由独立
`plan_capacity_bytes`记录，避免把长期占用写成“临时”。
