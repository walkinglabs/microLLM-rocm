# Experiment 069 — 同一个binary里，strict真的慢13%吗

Experiment 067把strict与uniform放在两个不同时段测量，DeepSeek T2048 B8端到端比值为
`0.866×`。但strict只改变一层Cache dtype，steady decode变化很小；13%可能是共享GPU漂移。

新runner不再调用PyTorch，也不换binary。每个shape运行三个新进程，奇数轮顺序
`uniform→strict`，偶数轮`strict→uniform`，然后分别取中位数。

72/72记录成功，12/12 suffix一致：

| 模型 | T | B | strict/uniform decode | prepare比 | E2E比 |
|---|---:|---:|---:|---:|---:|
| Qwen | 32 | 1 | 1.026× | 1.008× | 1.024× |
| Qwen | 32 | 8 | 0.973× | 0.891× | 0.962× |
| Qwen | 512 | 1 | 0.991× | 0.946× | 0.987× |
| Qwen | 512 | 8 | 1.003× | 0.997× | 1.002× |
| Qwen | 2048 | 1 | 1.003× | 1.023× | 1.005× |
| Qwen | 2048 | 8 | 1.014× | 1.036× | 1.030× |
| DeepSeek | 32 | 1 | 0.996× | 0.988× | 0.995× |
| DeepSeek | 32 | 8 | 0.991× | 1.002× | 0.992× |
| DeepSeek | 512 | 1 | 0.999× | 1.023× | 1.002× |
| DeepSeek | 512 | 8 | 1.040× | 1.007× | 1.030× |
| DeepSeek | 2048 | 1 | 1.001× | 0.999× | 1.001× |
| DeepSeek | 2048 | 8 | 1.034× | 0.994× | 1.011× |

![Same-binary KV policy](../assets/same-binary-kv-policy.svg)

关键DeepSeek T2048 B8不再有13.4%回退：prepare几乎相同，strict端到端反而高1.1%。
DeepSeek六点decode范围为`0.991×–1.040×`。Qwen T32 B8有3.8% E2E回退，但Qwen无需
strict策略，而且没有跨shape一致方向。

## 决定

保留Experiment 067 strict策略；**撤销“strict长batch稳定慢13.4%”的因果结论**。仍不自动
开启，因为layer 1来自固定checkpoint/prompt/误差门搜索，且Cache比uniform BF16多3.57%。

保留新的同-binary policy runner。今后小于10%的策略差异不能用跨时段summary相除，必须
使用该交替协议。这个节点改变的是证据质量，不是假装又找到一个Kernel加速。
