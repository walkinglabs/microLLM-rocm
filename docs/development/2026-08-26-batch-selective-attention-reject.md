# 为什么RMS变好仍然被拒绝

RMS像“平均偏差”，Max像“最坏一个位置”。batch-selective方案让平均偏差减少21.6%，但最坏偏差只
减少6.1%，还让B2变差。课程约定两者都至少改善10%，不能只展示好看的那一个。

速度方面四个batch都过门，最差0.994×；显存和allocation也不变。拒绝的唯一主要原因是完整数值证据
不够稳健，这正是correctness-first门存在的意义。

![Selective rejection](../../benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate/selective-gate.svg)
