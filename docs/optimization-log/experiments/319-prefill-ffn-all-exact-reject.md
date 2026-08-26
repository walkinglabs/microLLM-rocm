# Experiment 319 — Max改善35.5%，RMS却恶化5.8%

## 结论

all-batch exact gate/up的四个prefill为`1.000/0.983/0.964/0.991×`，性能门通过。完整logit Max从
0.001354降到0.000872，改善35.5%；RMS从0.0002294升到0.0002427，恶化5.8%。候选拒绝，FFN
vendor-solution模型线关闭。

## 这个失败说明什么

让gate/up在不同M下使用相同加法顺序，确实压低了最坏单点误差。但后续层的许多小误差总体能量反而
增大。Max和RMS回答不同问题，必须同时守门。

M8192 operator只有0.941×，但B4整模仍有0.964×。这说明局部性能失败不等于整模必然失败；不过数值
失败已经足以拒绝模型策略。

## 下一步

scope删除前只做一次诊断trace，验证exact gate/up是否把第一处差异移动到down。之后删除候选CLI、模型
scope和两个模型runner，不保留一条已经失败的用户路径。

原始结果见
[`benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate`](../../../benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate/README.md)。
