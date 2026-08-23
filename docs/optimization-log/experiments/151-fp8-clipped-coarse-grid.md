# Experiment 151：裁掉25%已经让RMS恶化6.5倍

修正runner后，fraction1的四个Max/RMS与Exp148逐值一致，证明pilot起点正确。随后只改变
activation amax fraction：

| fraction | worst RMS / F1 | worst Max / F1 | 全部top一致 |
|---:|---:|---:|---|
| 1.0 | 1.000× | 1.000× | 是 |
| 0.75 | 6.546× | 5.739× | 是 |
| 0.5 | 9.509× | 8.389× | 否 |
| 0.25 | 12.180× | 12.272× | 否 |

![Clipped activation coarse grid](../assets/fp8-clipped-coarse-grid.svg)

所有fraction仍是0/4完整精度门。0.75虽然没翻转top，但完整logits RMS已经灾难性恶化；0.5和
0.25连top也不稳定。每个clipped worker都精确记录Qwen 96/96、Deep 113/113 clipped/dynamic
调用，排除了“参数没有真正生效”。

## 当前结论边界

关闭`fraction≤0.75`，不再重复。coarse selector返回1.0，但1.0到0.75之间尚未采样；不能用四个
点声称所有clipping都失败。下一实验只测试0.95/0.9/0.85，并再次包含1.0 control。TPS仍不参与。
