# Matmul correctness-before-timing evidence

- `fp32-64-screen.json`：Readable与hipBLASLt完整输出都过门，7次Event P50/P95，只返回推荐；
- `fp32-128-accepted.json`：两候选过门，hipBLASLt被显式接受；
- `accepted-cache.jsonl`：接受后写出的exact environment持久entry；
- `fp16-strict-rejection.json`：零容差下hipBLASLt完整误差非零，因此Event/Wall四项时间全部保持0，
  Readable成为推荐；
- `verification.json`：机器可读边界。

这些数据只完成算子级correctness与重复计时。`accepted=true`是CLI显式动作，不代表仓库默认，
也不替代模型端到端回归。
