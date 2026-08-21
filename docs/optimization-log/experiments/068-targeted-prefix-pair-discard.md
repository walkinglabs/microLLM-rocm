# Experiment 068 — 只融合一个FP32层，长batch就会好吗

Experiment 066把所有层换成prefix-pair后失败。Experiment 067又发现strict策略的长batch
prepare较慢。新的、更窄假设是：只为那一个FP32敏感层删除per-head copy，uniform路径和
27个BF16层完全不动。

同一个Release binary提供`reference`与实验`paired-fp32`路由。DeepSeek T2048 B8、layer 1
FP32、其余BF16，2次warm-up、5次measured、3个新进程：

| 指标 | reference | paired | 结果 |
|---|---:|---:|---:|
| Cache prepare | 328.830ms | 333.866ms | 慢1.53% |
| End-to-end | 573.231ms | 576.605ms | 慢0.59% |
| Decode | 523.730 tok/s | 524.266 tok/s | 基本相同 |
| D2D calls | 4480 | 4320 | 少160 |
| D2D bytes | 2.433GB | 2.265GB | 少167.8MB |
| peak | 9.072GB | 9.072GB | 相同 |
| 16-token suffix | 固定 | 相同 | 通过 |

![Targeted prefix pair discarded](../assets/targeted-prefix-pair-discard.svg)

DeepSeek T512 B1完整logits仍为`max_abs 0.1851 / RMSE 0.03954`，说明候选没有破坏strict
精度。它也确实完成直接目标：只少的160次copy精确对应1个FP32层、B8、2个KV head、
Key/Value和5次measured prefill。

但主要prepare/E2E都没有改善。**discard**实验路由、Kernel、API和测试，保留Experiment
067 reference实现。这个同binary结果关闭“只要融合FP32层copy就能修好strict prefill”的
解释。下一步应转向allocator/lifecycle或请求调度，不再重试同类prefix copy融合。
