# Step 169 — official Qwen3 AdamW state audit

Status: complete

计划：复用正式checkpoint state；按310个参数命名两种moment；固定step/Max/RMS门；运行FP32/BF16；保存raw并删除大payload；生成SVG。

结果：FP32 620/620状态通过，BF16 Max失败。见[Experiment 384](../experiments/384-qwen3-training-adamw-state-audit.md)。
