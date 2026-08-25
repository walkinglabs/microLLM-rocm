# 2026-08-25 — 重新给训练画时间地图

旧地图来自更早的代码。如果道路已经修过，却仍拿旧地图决定下一步，很容易优化错误的地方。
因此本轮用当前二进制重新采集 Qwen 和 DeepSeek 的 B1T512 BF16 训练。

![Current training profile](../optimization-log/assets/current-training-profile.svg)

结果仍然很清楚：GEMM 占 58.56%/63.43%，AdamW 占 13.22%/18.16%。当前整步比旧
profile 快 1.0252×/1.0144×，但热点顺序没有变化。

这意味着下一轮不能因为 AdamW 看起来“也很大”就继续改阈值；这个方向已经做过完整
threshold 实验。新的任务应改变训练 GEMM 的组织方式，或建立能跨多个算子工作的稳定图。

所有结果都是 Kernel phase delta。它回答“GPU 时间去哪了”，不等于端到端训练吞吐提升。

