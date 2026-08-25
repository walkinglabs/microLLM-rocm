# Experiment 292：Softmax完全保序，P×V重排仍会让模型漂移

Status: model precision rejected

## 对照

current显式使用已保留的materialized exact-order路径；candidate显式关闭materialized并启用
exact-softmax split-P×V S16。DeepSeek T2048/B2/BF16/N64，三对fresh process交错顺序。

![Split-P×V model comparison](../../../benchmarks/results/2026-08-25-cached-attention-split-pv-model/comparison.svg)

| 指标 | current | split-P×V | 结论 |
|---|---:|---:|---|
| tokens/s中位 | 177.52 | 263.20 | 1.4834x |
| leave-one | — | 1.4829x–1.4856x | 速度稳定通过 |
| 64 token | — | 全相同 | top-1通过 |
| 完整logits Max/RMS | — | 0.064486/0.011488 | 精度失败 |
| peak delta | — | 0 | 不变 |
| KV bytes | 121,110,528 | 121,110,528 | 不变 |

三个pair的303,872 logits得到完全相同的Max/RMS失败。候选还增加17,920次逻辑allocation和65次
冷backend allocation；peak不变。

## 这次隔离告诉了什么

Experiment 291已证明score和softmax顺序不变，S1位级相同。因此本次0.064486漂移只需要P×V加法
树改变就能产生。即使operator context误差只有约1e-9，28层×64步仍会把它放大到完整分布不可
接受的程度。

速度和top-1不能覆盖完整logits门。模型路由拒绝，不跑Qwen边界矩阵，不改Auto。Step 109完成。
下一路线只考虑保持每个head逐position累加顺序的方案；第一项是跨GQA重复head复用同一value load，
它可以减少读取但不能改变每个head的加法顺序。

证据：[`split-P×V model gate`](../../../benchmarks/results/2026-08-25-cached-attention-split-pv-model/)
