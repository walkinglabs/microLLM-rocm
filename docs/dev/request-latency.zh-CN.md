# 吞吐更高，为什么单条请求可能更慢

Scheduler现在记录每条请求从提交到首token（TTFT），以及从提交到完成的wall time。GPU greedy选择
会把结果同步回host，所以时间包含真实prefill/decode完成，而不是只看CPU launch。

短上下文增加slot同时改善吞吐与TTFT：Qwen TTFT p50从135.1降到21.1ms，DeepSeek从226.6降到
28.0ms。长上下文不同：S4 TTFT p50最佳（55.9/100.4ms），S8吞吐最高但TTFT回升到64.0/
119.7ms，KV利用率只有46.85%。

因此选择slot要看目标：离线吞吐可偏S8；在线长请求median latency更适合S4。P95使用线性插值，
仓库同时保存每条请求原始数组，不能只展示一个分位数。

详见[Experiment 113](../optimization-log/experiments/113-request-latency.md)。
