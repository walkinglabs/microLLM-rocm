# Step 83 — Gradient producer out-kernel feasibility gate

Status: implemented, MI300 shape matrix pending

Direct leaf accumulation失败的原因已定位：producer仍申请普通gradient，随后再做leaf add。
下一节点不再做全模型开关，先选择一个真实、元素数大的parameter-gradient producer，增加调用方
提供output Tensor的独立算子路径，直接写最终地址。

门槛：CPU/HIP/PyTorch完整输出对齐；producer allocation少1、leaf add少1；Event与wall都至少
1.05×；同shape反例保留。只有operator门通过才允许重新进入Autograd小范围模型A/B；否则关闭
“预设leaf target但无producer out-kernel”路线，转向gradient-ready overlap。

实现新增`matmul_weight_gradient_out_`，它把rank-2 `input^T @ output_gradient`直接写入
caller-owned FP32 Tensor，复用已经验证的transpose-aware hipBLASLt `matmul_out_`。CPU、HIP、
PyTorch oracle对齐，地址与零payload transfer通过。runner覆盖Model-S head/FFN/Attention T32、
head T512和tiny反例，三次fresh process轮换shape与allocating/direct顺序。

Model-S head T32 pilot：完整3,145,728元素位级相同，logical allocation 1→0，Event/Wall
1.867×/1.581×。正式矩阵前不接Autograd。
