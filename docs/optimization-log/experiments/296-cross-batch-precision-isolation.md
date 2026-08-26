# Experiment 296：Batch漂移主要来自哪一个低精度岛

Status: BF16 FFN selected for block trace

## 四个策略只改变Linear权重准备

BF16 KV与materialized Auto保持不变。T2048 step0，B1/B2/B4/B8，每格两个fresh process：

![Precision isolation](../../../benchmarks/results/2026-08-25-deepseek-cross-batch-precision/precision-isolation.svg)

| 策略 | 最大Max | 最大RMS | 相对FP32 Max |
|---|---:|---:|---:|
| FP32 Linear | 0.001354 | 0.000229 | 1.0x |
| BF16 Attention only | 0.020970 | 0.004278 | 15.49x |
| BF16 FFN only | 0.062985 | 0.025171 | 46.54x |
| BF16 both | 0.067570 | 0.017350 | 49.92x |

32个进程全部确定，host/device argmax全部相同，step0 top1仍为151643。实际converted counts严格是
FP32 `(0,0)`、FFN `(84,0)`、Attention `(0,112)`、both `(84,112)`。

## 解释边界

FP32 Linear也有约1e-3 batch-shape漂移，说明通用GEMM算法变化是底噪。BF16 FFN-only把Max放大约
46.5倍且RMS最大，是主要放大源；Attention-only也有约15.5倍贡献。both不是两者简单相加，不能
从终点数值反推出每层因果。

## 决定

不改precision或scheduler。Step 114在cached step0给B1/B2启用block trace，先比较FP32 Linear与
BF16 FFN-only的28个block输出，找到第一个显著跃迁层；然后才深入该层的norm/gate/up/down。

证据：[`precision isolation`](../../../benchmarks/results/2026-08-25-deepseek-cross-batch-precision/)
