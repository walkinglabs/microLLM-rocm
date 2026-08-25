# Experiment 285：保留原归约顺序，还能不能加速

Status: bitwise operator candidate admitted to official model A/B

## 反驳方法

上一个split候选快2.22x却改变logits。新路径把逐position Q·K放进并行Kernel，写FP32 score Tensor；
finalize仍按旧fused完全相同的线程映射和顺序做max、exp、分母与P·V。

DeepSeek H12/KV2/D128覆盖T512/T2048、B1/B2和FP32/BF16 cache。8格各3个新进程，3次热身、
20次正式Event/wall测量。

![Materialized-score comparison](../../../benchmarks/results/2026-08-25-cached-attention-materialized-matrix/comparison.svg)

## 结果

| T | B | cache | current ms | materialized ms | Event | wall | score bytes |
|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 1 | BF16 | 0.0732 | 0.0561 | 1.303x | 1.260x | 24,576 |
| 512 | 1 | FP32 | 0.0969 | 0.0688 | 1.408x | 1.372x | 24,576 |
| 512 | 2 | BF16 | 0.0734 | 0.0565 | 1.298x | 1.249x | 49,152 |
| 512 | 2 | FP32 | 0.0962 | 0.0686 | 1.402x | 1.356x | 49,152 |
| 2048 | 1 | BF16 | 0.2733 | 0.1448 | 1.888x | 1.842x | 98,304 |
| 2048 | 1 | FP32 | 0.3696 | 0.1528 | 2.419x | 2.351x | 98,304 |
| 2048 | 2 | BF16 | 0.2729 | 0.1557 | 1.752x | 1.717x | 196,608 |
| 2048 | 2 | FP32 | 0.4191 | 0.1601 | 2.617x | 2.543x | 196,608 |

八格完整context全部与current逐元素位级相同。Event最小1.298x、wall最小1.249x；两次逻辑
allocation全部cache reuse，热backend allocation和payload transfer均为0。

## 决定

显式准入官方DeepSeek模型A/B，不改默认。该实验支持“QK grid并行度不足”并推翻“必须改变整个
softmax归约树才会变快”。下一门沿用Experiment 284的三对T2048/B2/BF16/N64协议，要求完整
logits位级相同、token相同、峰值/KV可解释和端到端至少1.05x。

证据：[`materialized-score matrix`](../../../benchmarks/results/2026-08-25-cached-attention-materialized-matrix/)
