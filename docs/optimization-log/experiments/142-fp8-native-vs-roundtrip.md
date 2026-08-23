# Experiment 142：原生FP8改变方向，但换FP32 GEMM也修不好

## 初中生版问题

Exp141像分别检查“尺子A”和“尺子B”。现在两把尺子都照常使用，只改变最后的乘法：

```text
full           = 两边FP8舍入 → 原生FP8 GEMM
both-roundtrip = 两边FP8舍入 → 还原FP32 → FP32 GEMM
```

两条路径的转换权重数、动态激活次数完全相同。这样，完整logits的直接差异才属于乘法路径。

## 两条不同的判断规则

“向量变化很大”不等于“离FP32更远”。本实验提前固定两条规则：

1. `full↔both RMS / full↔FP32 RMS ≥ 50%`：原生GEMM是主要额外向量扰动；
2. `full↔FP32 RMS / both↔FP32 RMS > 1.05`：原生GEMM显著增加最终总RMS。

| 模型/上下文 | direct/full RMS | full/both总RMS | 额外扰动大？ | 总误差显著增加？ |
|---|---:|---:|---|---|
| Qwen T8 | 58.05% | 1.002× | 是 | 否 |
| Qwen T512 | 75.45% | 0.765× | 是 | 否，full更低 |
| DeepSeek T8 | 76.91% | 0.832× | 是 | 否，full更低 |
| DeepSeek T512 | 54.81% | 0.998× | 是 | 否 |

![Native FP8 versus both roundtrip](../assets/fp8-native-vs-roundtrip.svg)

四个`full↔both`完整向量门都失败，说明原生GEMM的结果不是可忽略的小舍入差；但没有一组
满足“最终总RMS增加5%”。三个case的full RMS反而更低，证明误差方向会抵消，不能把direct
RMS当成可加到总误差上的一袋沙子。

## 决定

拒绝“把原生FP8 GEMM换成FP32 GEMM就能修好”的方案：both-roundtrip自己的四个FP32精度门
也全部失败。原生数学是重大扰动，却不是已证明的最终误差幅度主因。

下一步回到量化尺子本身。Exp141显示Qwen权重舍入占主导，而DeepSeek两边都重要；因此先实现
可用原生FP8 GEMM的per-output-channel权重scale，并用post-column scale避免当前outer-vector
运行时不支持的问题。任何新scale必须在真实`full`路径重新检查，不能只看软件反事实。
