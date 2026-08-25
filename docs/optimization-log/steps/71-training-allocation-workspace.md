# Step 71 — Training allocation/workspace attribution

Status: planned

Experiment 247 的候选每20步多1,920/2,240次逻辑分配，且Qwen长跑收益消失。下一步先用
allocation source和liveness证据判断临时cast/transpose/output是否能安全进入caller-owned
workspace；没有地址和生命周期证明前，不重开模型路由。

