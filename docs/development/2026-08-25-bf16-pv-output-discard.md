# 2026-08-25 — 这不是慢，而是库目前不接这道题

我们尝试让P×V矩阵乘法直接写BF16，这样可以不再为O projection单独转换。

但普通BTHD和GQA两种真实布局都被hipBLASLt以status 6拒绝，连计时都没有开始。

![BF16 P×V output discard](../optimization-log/assets/bf16-pv-output-discard.svg)

所以临时API已删除，不用“多试几个算法”伪装成还在推进。
