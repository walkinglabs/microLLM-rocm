# 2026-08-25 — ranked gather-scale result

![Ranked gather-scale discard](../optimization-log/assets/ranked-gather-scale-discard.svg)

gather-scale完成了计数目标，但没有改善已保留路线：相对本轮同步为1.0140x，候选8.778ms比
Step 102的8.687ms慢0.090ms，并增加1,368 bytes显存/描述传输。完整数值门通过。

性能路线拒绝，独立研究原语保留；Step 102继续作为显式T128最佳。当前ranked reducer局部
优化线停止，下一工作从新的端到端profile选择瓶颈。
