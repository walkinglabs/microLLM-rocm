# Step 99 — Ranked uneven-input weighting contract

Status: complete for tiny; explicit weighted mode kept

独立rank目前使用相同`B1×T`，所以简单gradient average等价于global-batch loss。真实最后一个batch
可能让各rank token数不同；继续平均每rank gradient会给小batch过高权重。

下一节点先不实现复杂join：

- 每step通信前交换rank-local有效token数；
- 默认要求全部相等，不相等时所有rank在梯度collective前有界失败；
- failure launcher必须终止peer而不是挂死；
- 提供显式weighted mode时，按`local_tokens/global_tokens`缩放再sum，和CPU拼接batch对齐；
- tiny手算与双rank参数门先通过，再考虑Model-S。

该合同与world4资源无关，可以在现有两卡环境继续推进。

实现已在训练前交换token count。tiny `[1,2]` equal-only两rank共同失败；token-weighted三步
rank exact、CPU Max/RMS `8.18e-8/8.79e-9`、loss差`1.94e-7`。weighted overlap仍明确拒绝。

正式结果复现全部门。tiny合同完成，Model-S同步weighted smoke移交Step100。
