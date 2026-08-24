# Step 44 — Device-owned AdamW step

Status: complete, explicit partial keep

## Decision

device step和两个bias correction让FP32/BF16 moment连续Graph重放与eager完整状态对齐。60进程
中FP32 64/256个1K Tensor为1.427×/1.436×；BF16与16×256K全部回退。保留显式原语，
默认关闭；下一步只测试稳定descriptor的两节点multi-tensor Graph。
