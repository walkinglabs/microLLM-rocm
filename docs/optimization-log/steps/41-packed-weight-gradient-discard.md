# Step 41 — Packed weight-gradient 反例

Status: complete, discard

## Decision

把pack D2D计入后，Qwen/DeepSeek QKV/gate-up四项为`0.835×–0.979×`，0/4过1.05门。
Autograd路由未建，weight-gradient组合搜索关闭。
