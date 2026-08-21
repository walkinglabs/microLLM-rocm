# Experiment 059 — cooperative RMSNorm weight-gradient columns

## 问题

Experiment 058 后最大的 Kernel 是 RMSNorm weight gradient：147 次共 `142.77 ms`。
旧实现让一个线程负责一个 hidden column，然后串行扫描512行。它与刚被淘汰的 softmax
单线程长行问题是同一种并行边界错误。

## 假设与反驳条件

当 rows≥256 时，让一个256-thread block负责一个 hidden column。每个线程处理部分 rows，
shared-memory reduction 汇总最终梯度。小 row count 保留旧 Kernel。

forward、input/weight gradient、零传输、两模型5%收益、显存和T128 fallback 任一失败即拒绝。

## 正确性

RMSNorm HIP 测试新增 rows=256，并与 width 16/384/512/896/1536 形成矩阵。每个 shape
同时检查 forward、input gradient、weight gradient和训练区零 host payload transfer。

## 正式结果

| 模型 | Experiment 058 | Experiment 059 | 自身加速 | measured peak | micro/PyTorch |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 2127.38 | 2594.81 tok/s | 1.220× | 不变 | 0.306× |
| DeepSeek Distill 1.5B | 1145.36 | 1288.95 tok/s | 1.125× | 不变 | 0.266× |

![Cooperative RMSNorm weight gradient](../assets/block-column-rmsnorm-weight-gradient.svg)

T128 走旧路径：`803.93→806.29 tok/s`（`1.003×`），peak相同。

## Profiler 与决定

目标 Kernel `142.77→8.72 ms`（`16.38×`）；全 Kernel `772.84→646.97 ms`
（`1.195×`），dispatch精确保持7631。保留。

新最大柱子是 AdamW `128.56 ms` 和 bias gradient `118.18 ms`。前者已有多次被拒绝的
向量化/分组证据；后者仍是单线程跨 rows reduction，下一反驳实验应先处理 bias gradient。
