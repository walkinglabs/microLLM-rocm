# 2026-08-25 — 长轨迹推翻短模型门

短跑看到Qwen/DeepSeek 1.0213×/1.0638×，20-step却看到Qwen只剩1.0006×。

![BF16 weight-gradient trajectory discard](../optimization-log/assets/bf16-weight-gradient-trajectory-discard.svg)

完整参数比较覆盖近9.8亿个值。两模型Parameter Max均超过预定门；Qwen的Parameter RMS和
loss轨迹也失败。五门只有峰值显存通过。

因此删除模型路由和两个候选runner。独立BF16算子仍有教学、算子研究价值；逐步loss输出和
通用safetensors完整比较也继续保留。下一问题是训练临时Storage能否通过caller-owned workspace
减少分配，而不是重新打开已经失败的低精度模型策略。

