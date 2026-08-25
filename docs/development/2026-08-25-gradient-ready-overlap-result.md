# 2026-08-25 — gradient-ready Event overlap结果

![Gradient-ready overlap](../optimization-log/assets/data-parallel-gradient-overlap.svg)

Model-S三策略A/B中，overlap相对同步views的total为1.0159×，finish wait从3.56降到1.55ms，
peak不变；45个loss、9次末步参数门和12个later overlap step全部通过。

它保持显式：相对transient仍多33,269,000B peak，而且单进程顺序backward不是标准DDP。
下一步进入one-process-per-GPU初始化、故障传播和独立rank训练路径。

发布门：CPU `365/365`、ASan/UBSan `363/363`、RCCL `36/36`、121个测试源。
