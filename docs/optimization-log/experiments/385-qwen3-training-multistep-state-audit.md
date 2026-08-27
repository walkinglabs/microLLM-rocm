# Experiment 385 — 从第一步扩到三步完整状态

Status: `FP32 pass; BF16 rejected`

![Qwen3 three-step state](../assets/qwen3-training-multistep-state-audit.svg)

FP32逐步loss、310最终参数、620最终moment和step=3共六门全过。BF16 loss、Parameter RMS、Moment Max失败，最坏状态传播到block2 FFN down。结论：FP32短轨迹成立；BF16不是单点误差。
