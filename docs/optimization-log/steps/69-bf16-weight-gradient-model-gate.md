# Step 69 — Gate/up-only BF16 weight-gradient model gate

Status: complete

只切换 gate/up weight gradient，query/KV永久保留FP32。三次交替顺序同二进制A/B，检查
48/56次真实路由、warm-up后首个loss与两步loss的相对差异、观察参数、峰值显存和
端到端tokens/s。首个measured loss发生在一次真实参数更新后，因此使用0.5%相对误差门，
不要求不可能的bit-exact。默认保持关闭，直到
模型门与后续更长训练轨迹均通过。

结果：Qwen/DeepSeek吞吐1.0213×/1.0638×，峰值不变，全部短门通过；候选继续显式。
