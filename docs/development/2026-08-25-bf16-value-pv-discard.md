# 2026-08-25 — 反方向的转换也不能直接交给库

保留BF16 V、其他部分不变的两种P×V布局都被库拒绝，没有开始计时。

![BF16 V P×V discard](../optimization-log/assets/bf16-value-pv-discard.svg)

现在剩余两个cast的库内mixed-dtype捷径都已用真实descriptor关闭。
