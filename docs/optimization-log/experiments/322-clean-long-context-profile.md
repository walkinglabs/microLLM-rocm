# Experiment 322 — 当前长上下文已经是1.139× PyTorch

## 当前端到端

三轮交替fresh process中位数：microLLM `177.77 tok/s`，PyTorch ROCm `156.04 tok/s`，比值
`1.1393×`。64个token完全一致；峰值5.23/6.38GB；KV都为121,110,528 bytes且100%利用。

旧0.8158×是materialized-score优化之前的历史基线，仍保留在旧实验中，但不能继续写成当前状态。

## 当前profile

1/3-generation phase delta：

| 类别 | 时间 | 占比 |
|---|---:|---:|
| cached Attention finalize | 346.92ms | 42.27% |
| hipBLASLt GEMM | 272.93ms | 33.25% |
| cached Attention scores | 64.64ms | 7.88% |
| FP32/BF16 cast | 33.74ms | 4.11% |

总Kernel 820.74ms，backend allocation delta为0。下一节点先审计是否存在未被五条失败/保留路线覆盖的
新finalize架构；没有结构差异就停止该局部线。

证据：

- [`clean baseline`](../../../benchmarks/results/2026-08-26-clean-deepseek-t2048/README.md)
- [`clean profile`](../../../benchmarks/results/2026-08-26-clean-deepseek-t2048-profile/README.md)
