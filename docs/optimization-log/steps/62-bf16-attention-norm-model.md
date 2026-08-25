# Step 62 — Attention Norm directly into QKV Arena

Status: complete, default enabled

## Decision

Qwen/DeepSeek整模为1.01309×/1.01303×，完整logits位级相同，allocation减120/140，
峰值减3.67/6.29 MB。BF16 QKV Arena默认启用，显式false保留。
