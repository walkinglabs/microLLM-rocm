# Qwen3通用shape矩阵与诚实状态

日期：2026-08-26
状态：执行覆盖通过，8个精度边界保留

通用官方推理runner现在消费Qwen3 stored/runtime双计数manifest，并覆盖context
1/32/128/512、batch 1/2、prefill、cached decode N1/N4/N32。

有效运行64/64 framework process成功。汇总后不是32/32答案通过，而是24 pass、8
`precision_mismatch`：prefill top token为8/8，decode完整token为16/24，KV active bytes为
24/24。固定mismatch列表和共同prefix在
[`benchmarks/results/2026-08-26-qwen3-fixture-shape-matrix`](../../benchmarks/results/2026-08-26-qwen3-fixture-shape-matrix/)。

基础设施同时补了三道门：

- manifest显式runtime count必须等于runner的兼容`parameter_count`；stored不能更小；
- worker都成功但token分叉时，row必须是`precision_mismatch`；
- 两个framework都看不到GPU时，整体是`invalid_environment`并立即停止剩余shape。

本矩阵只有1+1单进程，不提供稳定性能结论。下一节点定位第一处分叉的完整logits/层级，随后才决定
是精度policy差异、Cache积累还是实现错误。
