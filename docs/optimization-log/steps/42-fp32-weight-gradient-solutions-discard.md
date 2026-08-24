# Step 42 — FP32 weight-gradient exact solution

Status: complete, discard default

## Decision

Qwen/DeepSeek operator为`1.077×/1.133×`，exact registry命中144/168次；模型却为
`0.993×/0.996×`。默认与持久化均拒绝，显式研究seam保留。当前training GEMM
solution-index搜索关闭。
