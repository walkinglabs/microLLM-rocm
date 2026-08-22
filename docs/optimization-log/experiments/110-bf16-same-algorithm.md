# Experiment 110 — 75892让完整logits exact，代价1.3%–3.8%

共同solution `75892`在M32/M64均通过exact-shape support检查。3/3完整值pair的48个stage全部exact，
`first_nonzero_stage=null`。这确认默认不同algorithm是整个漂移链的原因。

![Same BF16 algorithm](../assets/bf16-same-algorithm.svg)

12个fresh无trace性能进程显示B1/B2分别为默认的0.9623×/0.9873×。因此保留版本局部、显式opt-in
registry，不把75892写成默认。下一步扩展更多FFN shape并研究自动选择“最快”或“跨batch严格”的
策略。

数据见[`110-data`](110-data/)。
