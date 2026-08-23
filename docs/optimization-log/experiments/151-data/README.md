# Experiment 151 data

这是修正weight minimum=0.005后的有效coarse activation-fraction pilot。

- 20 workers、16个完整logits comparison；
- fraction1四case与Exp148逐值一致；
- 0.75/0.5/0.25计数均证明所有dynamic activation被clipped；
- `fraction-summary.tsv`保存worst RMS/Max相对control；
- TPS明确不是证据；
- Exp149/150任何row均未读取或合并。

coarse选择1.0，但0.75到1.0之间仍需独立精细网格。
