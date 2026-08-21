# Experiment 075 — 取消请求后，Cache必须立刻归零

slot refill之前必须先定义“旧请求什么时候真正离开”。本实验为两个scheduler增加终态
`Cancelled`与幂等`cancel(id)`：第一次取消返回`true`，重复取消或已完成请求返回`false`，未知
ID抛出错误。

Reference scheduler在一个decode step后取消请求。测试先证明Cache字节大于0，再证明取消后为0；
已生成前缀保留，另一存活请求继续运行并与独立`generate()`一致。Admission scheduler在B3候选
中取消一行，剩余两行形成B2，取消行不进入CPU/HIP计算。

[`075-data/summary.json`](075-data/summary.json)记录本节点证据：CPU scheduler 6/6、HIP
scheduler 4/4；随后最终完整门为CPU 196/196、HIP 83/83、sanitizer 194/194。这个节点不宣称
加速，它只建立可验证的slot释放时刻。

决定：保留。后续continuous batching必须复用相同终态语义，不能让已取消slot继续占Cache。
