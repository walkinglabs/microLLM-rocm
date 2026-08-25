# Step 70 — Longer BF16 gate/up gradient trajectory

Status: complete, model route rejected

给训练 CLI 增加可选逐步 loss 记录，不在默认 JSON 中塞大数组。对 baseline/candidate 跑固定
种子、固定 T512 的更长轨迹，逐步比较 loss，并加入 gate/up 参数完整 Max/RMS 摘要。
只有训练轨迹和参数误差门都通过，才讨论默认策略；否则撤回模型路由但保留算子 API。

结果：五门仅峰值通过；Qwen长跑1.0006×，两模型Parameter Max失败。模型路由与候选runner撤回。
