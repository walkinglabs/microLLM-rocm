# Experiment 152 data

这是activation fraction精细网格，使用weight minimum0.005与retained O-only scope。

- 20 worker、16个完整logits comparison；
- fraction1四case与Exp148逐值一致；
- 0.95/0.9/0.85所有top稳定，但worst RMS均显著恶化；
- TPS不是证据；
- Exp149–151 row未读取或合并；Exp148仅用于fraction1 exact gate。

结合Exp151，0.25–0.95区间关闭，模型fraction保留1.0。
