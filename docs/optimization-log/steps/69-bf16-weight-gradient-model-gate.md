# Step 69 — Gate/up-only BF16 weight-gradient model gate

Status: in progress

只切换 gate/up weight gradient，query/KV永久保留FP32。三次交替顺序同二进制A/B，检查
48/56次真实路由、两步loss、观察参数、峰值显存和端到端tokens/s。默认保持关闭，直到
模型门与后续更长训练轨迹均通过。

