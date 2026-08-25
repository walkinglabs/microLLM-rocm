# Experiment 284：快2.22倍，为什么仍然不能默认开启

Status: performance passes; precision rejects model route

## 固定模型门

DeepSeek T2048/B2/BF16/S32/N64。current和split各3个新进程，奇偶轮换顺序；每进程warm-up 2，
measured 5次。每对导出303,872个cached logits和完整64-token suffix。

![Split model comparison](../../../benchmarks/results/2026-08-25-cached-attention-split-model/comparison.svg)

## 速度确实大幅改善

| 指标 | current | split | 比值/差值 |
|---|---:|---:|---:|
| 中位吞吐 | 133.27 tok/s | 297.02 tok/s | 2.2223x |
| paired speedup | — | 2.2286 / 2.2195 / 2.2223 | 全部稳定 |
| leave-one | — | 2.2209 / 2.2255 / 2.2240 | 全部过门 |
| engine peak | 5,231,076,352 | 5,231,076,352 | 0 |
| logical allocation | 184,815 | 202,735 | +17,920 |
| backend allocation | 95 | 96 | +1 |
| KV bytes | 121,110,528 | 121,110,528 | 0 |

split相对旧PyTorch 163.64 tok/s参考表面为1.815x，但这不能写成有效胜出，因为精度门先失败。

## token相同仍不是精度通过

三对64-token suffix逐项完全相同。然而每对完整logits都稳定得到：

```text
Max  = 0.0569113
RMS  = 0.0136965
```

这不是进程噪声，而是确定性的归约树变化。partial段分别做max/sum，再用log-sum-exp合并；单算子
context误差很小，经过28层与64步后被放大。仓库以前已拒绝“token相同但完整logits漂移”的候选，
这里不能降低标准迁就速度。

## 决定与反驳实验

模型默认拒绝，显式研究原语保留。下一候选只并行Q·K并物化每个position score；第二个finalize
Kernel严格复用当前fused的max、denominator和P·V顺序。它可能保留QK并行收益并恢复logits；也可能
因为global score流量而失去速度。

证据：[`split model gate`](../../../benchmarks/results/2026-08-25-cached-attention-split-model/)
