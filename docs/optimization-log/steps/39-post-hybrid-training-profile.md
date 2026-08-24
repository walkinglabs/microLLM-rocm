# Step 39 — Hybrid AdamW 后重新定位热点

Status: complete

## Decision

AdamW 每步时间相对 Experiment 213 提高 `1.372×/1.293×`，阈值搜索关闭。当前 GEMM 占
Qwen/DeepSeek Kernel 时间的 `59.33%/63.81%`，下一节点只能进入训练 GEMM 架构级变化。
