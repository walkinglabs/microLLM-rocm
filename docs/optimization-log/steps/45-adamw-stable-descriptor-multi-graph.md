# Step 45 — Stable-descriptor AdamW multi Graph

Status: complete, explicit keep

## Decision

descriptor在capture前一次上传，Graph固定为advance+multi update两节点。90进程完整状态对齐，
BF16 64/256个1K Tensor从0.767×/0.806×救到10.813×/36.929×；FP32 16×256K仍0.908×。
保留显式candidate，真实训练接入等待gradient Storage地址审计。
