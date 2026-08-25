# 2026-08-25 — allocation call 不等于新的显存申请

20-step候选记录中，Qwen多1,920次、DeepSeek多2,240次逻辑分配。除以route数量后，两者都
精确等于每route两次。

![BF16 weight-gradient allocation attribution](../optimization-log/assets/bf16-weight-gradient-allocation-attribution.svg)

字节数也能手算闭合：Qwen每route 5,898,240 bytes，DeepSeek 10,747,904 bytes，分别就是
input cast+transpose与dY cast两块BF16 Storage之和。

但backend allocation、峰值、cached bytes都没有增加，cache reuse增量与allocation call增量
完全相同。因此workspace可能只省少量host bookkeeping。下一实验先测成本，再决定接口。

对应 runner 会先预热 exact-size cache，再分开记录 Event 与同步 wall；同时要求公共API每次
恰好3次cache reuse、0次backend allocation。这样不会把首次显存申请误写成workspace收益。
