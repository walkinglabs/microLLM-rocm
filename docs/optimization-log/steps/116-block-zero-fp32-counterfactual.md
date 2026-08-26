# Step 116 — Block-0-only FP32 FFN counterfactual

Status: complete; precision policy rejected

Experiment 298证明第一处低精度放大发生在Block 0的FP32→BF16输入cast，gate和down继续放大。
这个节点只改变一件事：全局仍启用BF16 FFN，但第0层FFN保留FP32权重和激活。

固定DeepSeek、T2048、BF16 KV、materialized Attention、step0、B1/B2/B4/B8，各跑两个fresh
process，导出完整151,936 logits。比较：

- 当前全28层BF16 FFN；
- 仅Block 0保持FP32，其余27层BF16；
- 完整FP32 Linear参考。

必须报告converted tensor计数：当前84、Block-0 FP32应为81、全FP32为0。主要门是跨batch完整
logits Max/RMS和argmax；性能只作副指标，因为保留一层FP32可能增加时间和显存。

若Block-0 FP32显著降低最终漂移，进入前N层边界搜索；若改善很小，拒绝层选择策略并转向每层共享
cast/GEMM算法一致性。任何结果都不直接改变默认precision或scheduler。

结果：全局Max/RMS改善9.55%/42.86%，但B4/B8 Max变差12.7%/20.5%，B8 RMS也变差16.3%；
peak固定增加82,575,360 bytes，B8吞吐为0.9936x。策略拒绝。Step 117改查M=1/2/4/8共同
BF16 hipBLASLt solution，不再扩大FP32层数。
