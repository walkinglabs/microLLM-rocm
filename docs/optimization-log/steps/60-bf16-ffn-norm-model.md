# Step 60 — BF16 FFN Norm model route

Status: complete, default enabled with BF16 FFN Arena

## Decision

12进程中Qwen/DeepSeek为1.0122×/1.0092×，完整logits位级相同，峰值不变，测量
allocation减120/140。BF16 FFN Arena默认启用，显式false保留。
