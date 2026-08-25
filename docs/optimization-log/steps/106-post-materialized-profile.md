# Step 106 — Reprofile the retained T2048 default

Status: profile runner updated; measurement pending

Experiment 288保留了gfx942/BF16/known-head/uniform T>=2048自动路径。旧Experiment 281中cached
Attention占61.57%的profile已经过期。

下一实验使用同一DeepSeek T2048/B2/N64与load-subtracted 1/3-generation rocprof协议：

1. candidate不传materialized开关，必须报告`auto-enabled`；
2. 保存Kernel、HIP API、copy与allocation统计；
3. 做`(three-one)/2`得到一次generation；
4. 验证128个forward、64个token与新默认身份；
5. 选择新的最大Kernel类别，不先猜workspace或GEMM。

如果cached Attention仍第一，再分score/finalize；如果GEMM第一，回到shape/算法或更大图边界。任何
下一优化仍需完整logits/token和端到端门。
