# Experiment 286：完整logits位级相同，DeepSeek终于越过PyTorch吞吐参考

Status: explicit model route retained; broader default gate pending

## 同一个模型门重跑

仍是DeepSeek T2048/B2/BF16/N64，current与materialized各3个新进程，顺序交替，warm-up 2、
measured 5。唯一算法变化是candidate改为保序score物化。

![Materialized model comparison](../../../benchmarks/results/2026-08-25-cached-attention-materialized-model/comparison.svg)

## 结果

| 指标 | current | materialized | 结果 |
|---|---:|---:|---:|
| 吞吐中位 | 133.78 tok/s | 176.64 tok/s | 1.32068x |
| paired | — | 1.32068 / 1.32034 / 1.32071 | 全部稳定 |
| leave-one | — | 1.32053 / 1.32070 / 1.32051 | 全部过门 |
| 完整logits Max/RMS | — | 0 / 0 | 位级相同 |
| 64 token | — | 全部相同 | 通过 |
| peak bytes | 5,231,076,352 | 5,231,076,352 | 0 |
| logical allocations | 184,815 | 193,775 | +8,960 |
| backend allocations | 95 | 159 | +64 |
| KV bytes | 121,110,528 | 121,110,528 | 0 |

相对已固定PyTorch 163.64 tok/s参考为1.0794x。这个比较只适用于同一T2048/B2/N64环境；但因为
candidate完整输出与current位级相同，它没有Experiment 284的精度债务。

64次backend增量与64个不同prefix score尺寸一致，下一步可用最大capacity workspace反驳；不过
当前先不混入第二项修改。显式model route保留，自动默认仍等待Qwen/DeepSeek、T512/T2048、B1/B2
矩阵。

证据：[`materialized model gate`](../../../benchmarks/results/2026-08-25-cached-attention-materialized-model/)
