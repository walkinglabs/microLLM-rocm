# Step 119 — Block-0 prefill cache audit

Status: planned

Experiment 301排除了gate/up GEMM对相同BF16输入的跨M漂移。Experiment 298中decode当前Q/K/V投影
位级相同，但Attention context首先漂移；尚未检查的是full prefill已经写入的2048-token K/V前缀。

固定DeepSeek、T2048、BF16 KV、FP32 Attention投影、B1/B2/B4/B8、两个fresh process。只导出
Block 0 prefill后的完整key/value cache row 0，并报告：

- K与V各自的shape、dtype、元素数；
- B1与B2/B4/B8第0行Max/RMS/relative-L2/bitwise；
- 同一batch内部所有重复行是否位级相同；
- 两次process是否重复。

这不是性能运行。若cache前缀已漂移，继续拆prefill block0 hidden/norm/QKV；若cache exact，则构造
独立materialized cached Attention row-invariance测试。默认precision、algorithm和scheduler不变。
