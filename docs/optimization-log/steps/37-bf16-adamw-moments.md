# Step 37 — BF16 AdamW moment 状态

Status: complete, partial keep

## Decision

保留显式 BF16 moment 策略：两模型端到端提高 `1.0226×/1.0356×`，峰值降到
`0.8329×/0.8084×`，两份状态精确减半。默认仍是 FP32。

Qwen optimizer 只有 `1.0687×`，没有达到 `1.10×` stretch gate。全量 BF16 multi-tensor
pilot 被拒绝；下一步若继续，只能合并小 Tensor，并保留大 Tensor 的独立向量路径。
