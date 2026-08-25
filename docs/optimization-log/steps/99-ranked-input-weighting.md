# Step 99 — Ranked uneven-input weighting contract

Status: planned

独立rank目前使用相同`B1×T`，所以简单gradient average等价于global-batch loss。真实最后一个batch
可能让各rank token数不同；继续平均每rank gradient会给小batch过高权重。

下一节点先不实现复杂join：

- 每step通信前交换rank-local有效token数；
- 默认要求全部相等，不相等时所有rank在梯度collective前有界失败；
- failure launcher必须终止peer而不是挂死；
- 提供显式weighted mode时，按`local_tokens/global_tokens`缩放再sum，和CPU拼接batch对齐；
- tiny手算与双rank参数门先通过，再考虑Model-S。

该合同与world4资源无关，可以在现有两卡环境继续推进。
