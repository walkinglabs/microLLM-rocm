# Experiment 246 — 大算子变快以后，完整训练也变快了吗

Status: `short model gate passed; keep explicit`

同一二进制只切换 gate/up BF16 weight gradient。每个模型、每种策略三个新进程，第二组
反转顺序，避免慢漂移固定偏向候选。

| Model | Baseline tok/s | Candidate tok/s | Speedup | Peak | Routes |
|---|---:|---:|---:|---:|---:|
| Qwen | 15,631.52 | 15,964.51 | 1.0213× | 1.000× | 48 |
| DeepSeek | 6,500.38 | 6,915.39 | 1.0638× | 1.000× | 56 |

![BF16 weight-gradient model gate](../assets/bf16-weight-gradient-model.svg)

warm-up本身会更新一次参数，所以首个measured loss不应bit-exact。相对差异为
0.0712%/0.0088%；两步final loss差异为0.0201%/0.0035%，均低于0.5%门。
观察的非FFN参数相同，峰值显存不变，诊断精确命中48/56次且strided-copy为零。

候选每两步增加192/224次逻辑分配，说明下一步仍有workspace复用机会。但短跑不能证明
长期训练质量，因此开关继续默认关闭，只保留显式候选进入更长loss与参数轨迹。

本实验还捕获并修复了一个接线错误：最初候选只接到未被当前训练图调用的共享投影原语，
诊断稳定报告0次。最终实现用显式Linear role只标记FeedForward gate/up；down/QKV/O仍走FP32。

证据：[`model gate`](../../../benchmarks/results/2026-08-25-bf16-weight-gradient-model-gate/)

