# Experiment 130：少数token让整块Tensor的尺子失真

Exp129的Tensor amax仍失败。现在把每个T8输入按token row拆开，只诊断范围，不改变计算。

| 模型/边界 | row spread P50/max | row P50 / tensor max | ≤1/4范围的row |
|---|---:|---:|---:|
| Qwen attention norm | 3.11/6.34× | 0.367 | 18/192 |
| Qwen attention context | 1.60/20.43× | 0.794 | 4/192 |
| Qwen FFN norm | **4.03/61.76×** | **0.272** | **79/192 (41.1%)** |
| Qwen FFN activated | **4.82/1105.68×** | 0.660 | 37/192 |
| Deep attention norm | 1.56/2.39× | 0.798 | 0/224 |
| Deep attention context | 1.13/1.89× | 0.931 | 0/224 |
| Deep FFN norm | **3.84/7.03×** | **0.362** | 31/224 |
| Deep FFN activated | **3.93/2076.15×** | 0.550 | 43/224 |

![FP8 activation row range](../assets/fp8-activation-row-range.svg)

最清楚的反例：Qwen L2 activated的8个row amax为
`1701, 2.49, 2.99, 1.79, 1.54, 1.70, 1.63, 1.76`；Deep L2为
`3081, 3079, 2.62, 16.52, 2.54, 2.11, 4.01, 1.48`。Tensor scale被1–2个异常token决定。

结论不是“所有Linear改per-row”。证据支持先对`ffn_norm`和`ffn.activated`做FFN定向per-row；
Deep Attention几乎均匀，通用per-row只会增加reduction与scale应用成本。下一设计必须解决
hipBLASLt标量scale限制：row scale需要量化后在输出端逐row恢复，或使用库支持的vector scale；
没有正确输出缩放合同前不写Kernel。
