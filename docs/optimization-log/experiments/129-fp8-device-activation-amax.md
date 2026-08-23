# Experiment 129：误差再降八成，长context却慢二十倍

## 实现

每个Linear输入在GPU上执行：

```text
one-block amax
→ scale=max(amax/240, 0.0001)
→ FP8 quantize读取device scale
→ hipBLASLt或device-scale BF16 fallback
```

`ScaledTensor`允许device scale为真值、host float不可用。HIP门证明动态量化、反量化和prepared
Transformer热路径0 H2D/0 D2H。完整回归336/336通过，2个条件跳过；sanitizer定向6/6。

## 正式结果

| 模型/T | policy | TPS | FP8/BF16 | max/RMS | top | gate |
|---|---|---:|---:|---:|---|---|
| Qwen T8 | BF16 | 2396 | — | 0.092/0.0165 | equal | pass |
| Qwen T8 | FP8 dynamic | 1504 | 0.627× | 0.999/0.192 | equal | fail |
| Qwen T512 | BF16 | 92391 | — | 0.105/0.0160 | equal | pass |
| Qwen T512 | FP8 dynamic | 4874 | **0.0527×** | 1.475/0.293 | equal | fail |
| Deep T8 | BF16 | 1373 | — | 0.047/0.0083 | equal | pass |
| Deep T8 | FP8 dynamic | 844 | 0.614× | 1.243/0.438 | equal | fail |
| Deep T512 | BF16 | 49541 | — | 0.045/0.0087 | equal | pass |
| Deep T512 | FP8 dynamic | 2181 | **0.0440×** | 1.240/0.249 | equal | fail |

![FP8 device activation amax](../assets/fp8-device-activation-amax.svg)

## 数值上改善了多少

相对Experiment 127的固定activation 0.2，RMS下降：

```text
Qwen T8     0.667 → 0.192  (-71%)
Qwen T512   1.298 → 0.293  (-77%)
Deep T8     1.175 → 0.438  (-63%)
Deep T512   1.309 → 0.249  (-81%)
```

但它们仍为0.05门的3.85×、5.85×、8.76×、4.98×，四个精度门全失败。

## 为什么长context极慢

第一版让一个256-thread block扫描完整输入。T8数据少，已经有37%–39%吞吐损失；T512每个线程
串行读取大量元素，再乘以每层多个Linear，Qwen/Deep相对BF16慢19×/23×。这不是MI300 FP8
硬件慢，而是我们的reduction没有并行扩展。

## 决策

保留device-scale合同、动态量化和fallback，因为数值改善且无host同步；拒绝当前模型策略和
single-block reduction。下一步先测同一Tensor内部不同row/token的amax分布，决定per-row还是
per-token；同时多block reduction是明确性能债，但只有下一数值策略复用它时才优化，避免优化将被
删除的错误粒度。
