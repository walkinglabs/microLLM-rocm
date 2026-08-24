# 扩大 Grouped QKV 搜索，并把首次准备时间说清楚

日期：2026-08-24

前一轮只检查16个算法，DeepSeek整模只快0.15%。扩大到64个以后，Qwen/DeepSeek稳定选出
`64713/64755`，算子快2.010×/1.692×，整模快4.58%/2.95%。这说明“GroupedGemm不适合
DeepSeek”的解释被推翻，真正的问题是候选覆盖不足。

但又出现了新的现实问题：GroupedGemm第一次初始化很贵。最初每个block初始化一次，首次forward
约5.7秒。后来改成一个共享kernel加device user arguments，每个block只上传588字节左右的参数，
总参数准备不到0.7ms。这样只剩一次kernel初始化，但正式中位数仍约204–208ms。

所以报告分两种速度：

- 长驻服务已经预热：Qwen 1.0458×，DeepSeek 1.0295×，可以显式使用；
- 一次性程序首次请求：多约0.2秒准备，没过100ms门，不能默认打开。

完整logits、top token和峰值都通过。这里最重要的不是“最终快了”，而是没有用warm-up把首次成本
藏掉。未来 serving scheduler 若能在接收请求前预热，可以再测真正的TTFT；当前CLI默认仍走旧路径。
