# Experiment 300：同一个solution index，不代表同一棵加法树

Status: solution 75892 rejected for decode

## 先证明候选真的共同支持

DeepSeek gate/up固定`K=1536,N=8960`、BF16输出和32 MiB workspace。M1/2/4/8各请求64个heuristic，
64个index全部在交集中。历史prefill实验用过的75892在四个shape都支持，workspace都是4,587,520 bytes。

然后固定T2048、BF16 KV、materialized Attention、全BF16 FFN，比较默认dispatch和四个M都显式注册
75892。B1/2/4/8、两次fresh process，共16进程。

![BF16 decode algorithm](../../../benchmarks/results/2026-08-26-deepseek-bf16-decode-algorithm/algorithm.svg)

| 指标 | 默认 | 75892 | 比率 |
|---|---:|---:|---:|
| 全局最大Max | 0.062985 | 0.069939 | 1.1104x |
| 全局最大RMS | 0.025171 | 0.022978 | 0.9129x |
| B4 RMS | 0.012237 | 0.019656 | 1.6063x |
| B8 RMS | 0.012372 | 0.018160 | 1.4678x |
| 最低吞吐比 | — | — | 0.9853x |
| peak增量 | — | 4,587,520 bytes | 等于workspace |

全局RMS表面变好只是B2主导最大值；B4/B8反而明显恶化。16进程重复和argmax全部通过，但相同行仍不
位级相同。即使index相同，hipBLASLt内部仍可能按M选择不同tile/归约细节；而完整模型还包含Attention
context与down projection，因此不能把“共同支持”写成“共同数值顺序”。

## 决定

75892 decode拒绝，不写默认。下一步在算子层构造一行BF16输入并重复到M1/2/4/8，对64个共同候选
逐一比较第0行完整8960输出。先找真正row-invariant的solution，再让通过者进入完整模型；若0/64通过，
固定vendor solution路线关闭，转向自有保序Kernel或允许容差的批处理语义。

证据：[`BF16 decode algorithm`](../../../benchmarks/results/2026-08-26-deepseek-bf16-decode-algorithm/)
