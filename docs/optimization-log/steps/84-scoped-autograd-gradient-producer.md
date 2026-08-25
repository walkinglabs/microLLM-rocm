# Step 84 — Scoped Autograd caller-owned weight-gradient producer

Status: planned

Operator门已经证明caller-owned weight-gradient producer在5/5 shape同时提高Event/Wall并删除
一次allocation。下一步只改rank-2 matmul的right leaf backward：当leaf被显式标记为“零初始化、
尚无贡献的accumulation target”时，producer直接覆盖目标；一旦已有贡献、别名、非leaf、非连续
或shape不匹配，必须恢复普通Tensor producer+accumulate。

先做CPU分叉/重复/预置非零拒绝与HIP地址/transfer门，再做独立Autograd micro A/B。只有完整
right gradient、left gradient、loss和地址全部通过，且Autograd wall/Event≥1.05×，才允许考虑
Model-S某个untied线性权重；DDP route继续不存在。
