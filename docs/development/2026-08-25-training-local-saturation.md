# 2026-08-25 — 停止调训练小开关

当前训练时间主要在GEMM和AdamW。剩余cast即使瞬间完成，Qwen/DeepSeek Kernel也最多改善
1.0332×/1.0277×。

![Training local saturation](../optimization-log/assets/training-local-saturation.svg)

六条相邻路线已经有完整反例。继续换阈值、solution index或小workspace会重复已有实验。
下一阶段应做新的custom kernel/graph-wide设计，或推进真实data-parallel reducer。

“局部饱和”不是“框架完成”。它只说明问题尺度必须升级。

