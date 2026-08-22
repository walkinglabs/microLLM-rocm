# Experiment 111 — Qwen index 75789近乎免费，但不exact

Qwen M32/M64 FFN各返回64候选，共同index 56个。首选共同index `75789` 通过shape support，但3/3
完整值pair仍从block0 gate开始分叉：gate/up max均0.00390625，最终logits max0.083515。

![Qwen common algorithm discard](../assets/qwen-common-algorithm-discard.svg)

12条无trace性能中，B1为默认0.9932×，B2为1.00045×；B2有一次偏高样本，不能解释成收益。
因此75789被拒绝为Qwen strict-exact策略。Qwen默认token本已跨slot一致；若追求tensor exact，下一
节点必须扫描更多共同候选，而不是回退全部FFN。

数据见[`111-data`](111-data/)。
