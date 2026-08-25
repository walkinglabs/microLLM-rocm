# 2026-08-25 — 为什么低精度梯度只能挑 shape

相同算法在不同矩阵大小上可能完全相反。BF16 weight-gradient 候选必须先转换两个输入；
小矩阵省下的计算不够支付转换成本，大矩阵才可能获益。

![BF16 weight-gradient shapes](../optimization-log/assets/bf16-weight-gradient-shapes.svg)

Qwen/DeepSeek gate/up 分别达到 1.459×/1.890×，query/KV 却只有 0.718×–0.976×。
所以框架没有新增全局 low-precision gradient 开关。正式 API 负责表达清楚的 BF16 数学，
Autograd 研究开关只允许 gate/up 使用，并且默认关闭。

CPU、HIP、PyTorch 三方都按“两个输入先舍入为 BF16，再用 FP32 累加/输出”对齐。它与
FP32 梯度本来就不应 bit-exact；完整 Max/RMS 必须继续报告，并由模型 loss/训练轨迹决定
这种精度变化是否可接受。

