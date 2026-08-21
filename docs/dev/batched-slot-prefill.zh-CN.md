# Batched slot prefill：新同学一起写第一段草稿

Continuous scheduler以前即使8个prompt长度相同，也逐个调用8次B1 prefill。decode已经batch化，入场
阶段却仍串行，所以uniform continuous远慢于static batch。

新模型接口：

```cpp
forward_prefill_cached_rows(tokens[A,T], shared_cache, target_rows[A]);
```

它先用一个临时A-row Cache运行一次完整`[A,T]` prefill，再把每行K/V映射进共享Cache的空slot。
已有请求的row、position和Storage地址不变。

Scheduler使用稳定分组：先看最早pending请求的prompt长度，再从队列中按提交顺序收集同长度请求，
最多填满当前空slot。不同长度不会被错误padding进同一prefill；它们会在同一个scheduler step中进入
后续group。

Release结果说明可合批行数决定收益：

- uniform R8/S8：8行一次prefill，2.931×baseline，达到static的87.4%；
- divergent R8/S4：6行进入3个batch，1.313×；
- divergent R8/S2：只有2行能合批，1.056×。

吞吐提高也可能让更多请求同时驻留，因此active KV峰值可以增加；allocated Cache仍由固定slots决定，
这不是泄漏。

实验见 [Experiment 101](../optimization-log/experiments/101-batched-slot-prefill.md)。
