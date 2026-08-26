# Experiment 325 — decode rows2 grouped gate/up值得进入模型门

当前DeepSeek B2 decode每步只有2行，旧矩阵只覆盖rows256/512/1024。三轮probe中solution65193稳定，
64/64候选exact，Event 1.814×、wall 1.519×。Qwen index不稳定，不进入route。下一步只扩展已有Arena
grouped机制到cached decode，并跑固定DeepSeek完整模型。

![Rows2 grouped](../../../benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2/grouped-row2.svg)
