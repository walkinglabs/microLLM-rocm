# T2048 Prefill Attention 的两种首差

## 初中生版本

Attention有三步：先打相关分数，再把分数变成比例，最后按比例取回内容。

我们原本猜“所有batch都在同一步出错”。完整检查后发现不是：

- 两个请求一起算时，分数和比例完全一样，最后取内容的P×V先不同；
- 四个或八个请求一起算时，最早的QK分数已经不同。

这像同一条流水线用了两套机器配置。两件产品时，最后一台机器改变了加法顺序；四件和八件时，
第一台机器就换了计算顺序。因此不能只修最后一台机器。

## 怎样避免误判

QK会计算“未来位置”的分数，随后被causal mask丢掉。raw score第一个差异恰好在未来区，所以我们又
只检查真正可见的下三角。B4/B8的第一个可见差异仍存在于索引2048，证明QK差异真的会进入softmax。

每个B1行检查50,331,648个score和probability，其中causal可见值25,178,112个；P×V检查
3,145,728个输出。两轮process结果一致，大文件比较后全部删除。

## 下一步

分别筛QK和P×V的hipBLASLt solution。必须先做完整数值门，再计时，再进完整模型；softmax目前只传播
上游差异，不先改它。

![Complete Attention core matrix](../../benchmarks/results/2026-08-26-prefill-attention-core-matrix/attention-core.svg)
