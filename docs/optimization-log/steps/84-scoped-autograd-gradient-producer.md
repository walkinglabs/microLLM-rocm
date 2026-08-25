# Step 84 — Scoped Autograd caller-owned weight-gradient producer

Status: complete, discarded and removed

Operator门已经证明caller-owned weight-gradient producer在5/5 shape同时提高Event/Wall并删除
一次allocation。下一步只改rank-2 matmul的right leaf backward：当leaf被显式标记为“零初始化、
尚无贡献的accumulation target”时，producer直接覆盖目标；一旦已有贡献、别名、非leaf、非连续
或shape不匹配，必须恢复普通Tensor producer+accumulate。

先做CPU分叉/重复/预置非零拒绝与HIP地址/transfer门，再做独立Autograd micro A/B。只有完整
right gradient、left gradient、loss和地址全部通过，且Autograd wall/Event≥1.05×，才允许考虑
Model-S某个untied线性权重；DDP route继续不存在。

实现提供overwrite-only target：producer成功时完整覆盖且保持地址；任何generic first
contribution都会放弃target并恢复普通assignment，绝不读取未初始化内容。默认关闭的rank-2
right-weight dispatch记录调用数。CPU覆盖普通/非零回退、generic fallback和shared weight首个直写；
HIP覆盖左右梯度、地址和零payload transfer。

Autograd backward-only runner复用已构建graph，覆盖与operator相同五shape并轮换顺序。pilot显示
虽然allocation少1，FFN T32为0.960×/0.976×、head T32为0.983×/0.985×；正式矩阵仍需确认。

正式结果：5/5 gradient exact且地址保持，但Event 0.976×–1.035×、Wall 0.991×–1.018×，
0/5过1.05门。scoped dispatch和target状态API撤回；caller-owned operator保留。
