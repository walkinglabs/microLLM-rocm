# Experiment 320 — gate/up/SwiGLU全Exact以后，down首差

## 结论

exact Attention和exact gate/up控制下，FFN norm、gate、up、SwiGLU activation在B2/B4/B8跨batch与
同batch重复行全部位级一致。down统一成为第一处差异，跨batch Max为`1.72e-5/1.05e-5/1.43e-5`。

## 意义

这证明上一阶段的gate/up scope确实改变了预期边界，不是runner误标。但Experiment 319已经证明该模型
策略RMS恶化，所以因果正确不等于默认可用。

接下来先删除失败模型route，再单独筛down的K8960/N1536 descriptor。只有down operator同时exact且
每M过性能门，才允许重新增加一个不同的、只属于down的实验scope。

原始统计见
[`benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace`](../../../benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace/README.md)。
