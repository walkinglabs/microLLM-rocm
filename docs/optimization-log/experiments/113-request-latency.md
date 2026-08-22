# Experiment 113 — 长context的S4 TTFT优于S8

48/48 fresh processes通过。短请求S8同时获得最高吞吐和最低TTFT；长请求S8吞吐最高，但Qwen/
DeepSeek TTFT p50从S4的55.9/100.4ms回升到64.0/119.7ms，completion p50也回升。

![Request latency](../assets/request-latency.svg)

S8 long KV byte利用率46.85%、slot利用率75%。这说明预留更多最大长度slot会改善并行吞吐，却不
保证单请求median latency继续下降。DeepSeek long S8还有一次216 tok/s低值，已保留。

策略结论：short优先S8；long在线服务先以S4为默认，吞吐批处理才选择S8。数据见[`113-data`](113-data/)。
