# Step 94 — Ranked overlap scale boundary

Status: complete, context-selective overlap retained

Experiment 270在Model-S T32上拒绝overlap：可隐藏的通信尾部小于hook/Event/enqueue成本。下一
节点不再微调T32实现，而是建立独立context尺度track。

worker需要显式`--context`并生成相同规则的rank-local/global CPU batch；至少比较T32与T128，
资源允许再加T512。每个context先只做同步views/overlap views，完整参数参考可按相同初始状态
复用，不能把CPU重复运行计入GPU指标。

新track必须报告compute window、finish wait、total、current/peak与数值门。只有更长context下
overlap稳定过1.01，才说明它是尺度选择策略；否则关闭当前Model-S overlap路线并转向checkpoint
ownership或更大真实模型的分布式训练。

实现新增显式context合同与专用T32/T128 runner。第一次pilot拒绝了“跨context peak相同”的错误
测试假设，保留同context memory门；数值门未变。单次T32 total `1.039×`、T128 `0.959×`，
finish均约2×，current/peak增量0。正式三次前不关闭尺度track。

正式结果：T32 total `0.9995×`，T128 `1.0923×`；删除最慢run后T128仍`1.069×`。两尺度
显存增量0、完整数值门通过。保留显式context策略：T32同步、T128 overlap；一般默认仍关闭。
