# Experiment 206 — 直接 BF16 Q/K 的 sequence/batch 矩阵

Status: keep explicit policy across measured shapes; default remains off

## 问题

Experiment 205只证明B1/T512。GroupedGemm的`rows`相同并不代表workload相同：B1/T1024与
B2/T512都使用rows1024，但Attention序列长度、batch布局和last-logit导出完全不同。本实验把
三种case分开，并对每个batch行检查完整词表输出。

## 正式矩阵

![BTHD BF16 Q/K shape matrix](../assets/inference-bthd-bf16-qk-shapes.svg)

MI300X/gfx942，每个策略五个独立进程，2次热身、5次计时：

| Model | Case | FP32边界 | BF16 Q/K | 加速 | Logits/rows | Peak |
|---|---|---:|---:|---:|---:|---:|
| Qwen | B1/T256 | 76,739 | 78,610 | 1.0244× | bit-exact | 不变 |
| Qwen | B1/T1024 | 125,896 | 127,633 | 1.0138× | bit-exact | 不变 |
| Qwen | B2/T512 | 152,932 | 154,883 | 1.0128× | 2行bit-exact | 不变 |
| DeepSeek | B1/T256 | 40,106 | 40,861 | 1.0188× | bit-exact | 不变 |
| DeepSeek | B1/T1024 | 68,935 | 69,983 | 1.0152× | bit-exact | 不变 |
| DeepSeek | B2/T512 | 74,101 | 75,542 | 1.0194× | 2行bit-exact | 不变 |

所有候选进程的retained计数都等于`block_count × 7 forwards`，所有控制进程为0。

## 反例与重复

三进程pilot中，Qwen B2/T512只有1.0091×，未通过固定1.01门；其他五项为
1.0129×–1.0233×。我们没有降低门，而是按Exp205的稳定性规则扩大到五进程。正式结果六项
全部通过。Pilot原始数据继续保留，提醒后续约1%的候选需要更强重复次数。

## 决策

在已测sequence/batch上保留显式策略。默认仍关闭，因为证据只覆盖一个gfx942/hipBLASLt版本
和固定模型。该精度边界的shape扩展已经完成；下一优化候选应回到Experiment 204的另一个热点
causal softmax，而不是继续扩大相同cast策略。

原始证据：

- [三进程pilot](../../../benchmarks/results/2026-08-24-inference-bthd-bf16-qk-shapes-pilot3/)
- [五进程正式矩阵](../../../benchmarks/results/2026-08-24-inference-bthd-bf16-qk-shapes-formal5/)
