# Experiment 291：不动Softmax，只拆P×V，速度和误差会怎样

Status: operator admitted to official-model gate

## 隔离变量

先前split-sequence同时改变score、softmax和P×V归约，模型虽然快却出现0.05691 logits Max误差。
本实验保留并行score和原256-lane softmax，只把P×V按连续position片段拆开。

![Split P×V search](../../../benchmarks/results/2026-08-25-cached-attention-split-pv-matrix/split-pv-search.svg)

两模型、T512/T2048、B1/B2、FP32/BF16，S1/2/4/8/16各两个fresh process，共160条。

| 证据 | 结果 |
|---|---:|
| S1位级等于materialized current | 16/16 |
| S1是性能反例 | 16/16 |
| winner通过完整context与双性能门 | 16/16 |
| winner | 全部S16 |
| Event speedup | 1.2749x–2.9549x |
| wall speedup | 1.2372x–2.6373x |
| 最大context Max/RMS | 3.90e-9 / 1.09e-9 |

目标DeepSeek T2048/B2/BF16的Event/wall为2.2908x/2.1372x，probability和partial各196,608
bytes，热backend allocation为0。

## 当前能说什么

S1慢说明新增两个Tensor和launch本身是成本；S4以后变快说明P×V序列并行覆盖了成本。保留exact
softmax后，context误差仍是1e-9级，也显著小于先前全split路径。

但28层×64步可能继续放大这点误差。算子通过只允许进入三对DeepSeek完整logits门，不允许直接
写模型默认。Step 109会比较materialized current与S16的303,872 logits、64 token、吞吐、peak、
KV bytes和allocation。

证据：[`split-PV matrix`](../../../benchmarks/results/2026-08-25-cached-attention-split-pv-matrix/)
