# 2026-08-25 — ranked weighted-overlap result

![Ranked weighted overlap discard](../optimization-log/assets/ranked-weighted-overlap-discard.svg)

Model-S `[B1,B2]` T128的ready overlap在数学上正确，但当前实现更慢。finish wait缩短1.930x，
57次leaf scale却给forward/backward增加1.520ms，steady step最终为0.9594x。

三轮策略间完整参数比较均为0/0，CPU Max/RMS为`0.004938 / 3.218e-6`，显存增量0。性能
路由拒绝，正确性原语保留。下一节点只尝试把57次leaf scale合并成3次bucket scale。
