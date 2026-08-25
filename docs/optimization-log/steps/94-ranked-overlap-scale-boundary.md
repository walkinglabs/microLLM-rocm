# Step 94 — Ranked overlap scale boundary

Status: planned

Experiment 270在Model-S T32上拒绝overlap：可隐藏的通信尾部小于hook/Event/enqueue成本。下一
节点不再微调T32实现，而是建立独立context尺度track。

worker需要显式`--context`并生成相同规则的rank-local/global CPU batch；至少比较T32与T128，
资源允许再加T512。每个context先只做同步views/overlap views，完整参数参考可按相同初始状态
复用，不能把CPU重复运行计入GPU指标。

新track必须报告compute window、finish wait、total、current/peak与数值门。只有更长context下
overlap稳定过1.01，才说明它是尺度选择策略；否则关闭当前Model-S overlap路线并转向checkpoint
ownership或更大真实模型的分布式训练。
