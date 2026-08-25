# Step 107 — Exact-order finalize thread mapping

Status: planned

Experiment 289证明DeepSeek T2048/B2/N64中，保序finalize单独占349.17ms/42.00%，大于GEMM。

本节点只改变finalize内部工作分配：

1. 保持输入score、BF16/FP32 value、输出和公开API不变；
2. 候选不得改变position累加顺序，先争取位级相同；
3. 固定H12/KV2/D128与H14/KV2/D64，覆盖T512/T2048、B1/B2和两种cache dtype；
4. 每格fresh process、热身和Event/wall重复测量；
5. 保存完整context、allocation和候选选择，不凭单次最快值；
6. 只有operator winner通过DeepSeek T2048完整logits/token/peak/端到端门，才考虑模型路由。

第一组可反驳候选是`blockDim=64/128/256`。当前width为64或128，256线程有大量线程不参与P×V，
但更大的block可能加快max/denominator；较小block提高每CU驻留，却会改变block reduction的树。
因此“位级相同”与“数值容差内”必须分开报告，不能先假定哪一个更快或更准。
