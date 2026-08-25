# Step 113 — Cross-batch precision-island isolation

Status: planned

Experiment 295证明DeepSeek batch shape在step0就产生完整logits漂移，同时排除行索引和argmax。

固定T2048、step0、B1/B2/B4/B8、每格两个fresh process，比较：

1. `fp32`：FFN/Attention均不准备BF16；
2. `bf16-ffn`：只准备FFN；
3. `bf16-attention`：只准备Attention；
4. `bf16-both`：当前路径。

每个策略保存：

- B1对各batch完整151,936 logits Max/RMS；
- batch内部行与跨进程位级门；
- host/device argmax；
- 实际converted tensor与policy字段；
- peak、运行时间只作诊断，不做性能选择。

若FP32仍有同等级漂移，下一步trace通用batch GEMM/Norm；若某个BF16 island首次放大，则对该island
做逐层输出trace。任何修复必须同时保持当前B2性能和Experiment 288默认收益。
