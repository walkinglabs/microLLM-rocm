# Step 71 — Training allocation/workspace attribution

Status: complete

Experiment 247 的候选每20步多1,920/2,240次逻辑分配，且Qwen长跑收益消失。下一步先用
allocation source和liveness证据判断临时cast/transpose/output是否能安全进入caller-owned
workspace；没有地址和生命周期证明前，不重开模型路由。

结果：每route恰好多两次cast Storage分配，字节恒等；backend/peak/cached增量均为0。
