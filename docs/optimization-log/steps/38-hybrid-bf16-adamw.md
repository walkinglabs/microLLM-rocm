# Step 38 — 小 Tensor 合并，大 Tensor 保留带宽路径

Status: complete, keep

## Decision

BF16 moment 的 HIP Auto 使用 1,048,576-element 阈值。五进程 Qwen/DeepSeek optimizer
`1.240×/1.263×`，端到端 `1.049×/1.053×`，全部门通过。

16M 反例使 DeepSeek optimizer/E2E 变成 `0.896×/0.980×`。因此不允许把“合并更多”写成
单调优化，也不把 1M 推广到其他 GPU。
