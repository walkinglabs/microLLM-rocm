# Step 120 — Block-0 full-prefill trace

Status: planned

Experiment 302证明Block 0 BF16 K/V前缀在decode前已经随batch漂移。这个节点只增加full-prefill
可观测性，不改变数学或默认策略。

固定DeepSeek、T2048、FP32 Linear、BF16 KV、B1/B2/B4/B8、两个fresh process，记录：

- embedding与Block 0 input；
- attention norm；
- FP32 Q/K/V projection；
- Q/K RoPE与current value；
- 转成BF16并写入后的packed K/V；
- Block 0 attention context/output（用于闭合，不先优化）。

每个边界比较第0行完整值和batch内部重复行。若Q/K/V projection已经不同，进入大M FP32 GEMM
row-invariance/solution审计；若projection exact而BF16 packed cache不同，检查cast/store。trace不做性能声明。
