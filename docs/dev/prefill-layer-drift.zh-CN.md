# 从第0层到完整词表：B1/B2数值差异怎样长大

前一实验已经排除B2 row和KV copy错误。现在的问题变成：同一个prompt从B1变成B2以后，数值从
哪一层开始不同，最后相差多少？

## 1. 怎样比较

模型权重、P5的32个token、dtype和GPU全部相同：

```text
B1: [P5]
B2: [P5, P5]
```

TraceSession现在也覆盖graph-free inference。诊断保存：

- embedding输出；
- 28个Transformer block输出；
- final RMSNorm输出；
- 151936维完整last-token logits。

每个阶段同时做两个比较：B1与B2 row0，以及B2 row0与row1。临时完整tensor只用于本次诊断，
仓库保存压缩后的max/mean/RMS/relative-L2；这些host snapshot不参与性能结论。

## 2. 第一个差异在哪里

Embedding逐值完全相同，所以token读取和batch row布局没有问题。第一个非零差异出现在block 0：

| stage | max-abs | mean-abs | RMS | relative-L2 |
|---|---:|---:|---:|---:|
| embedding | 0 | 0 | 0 | 0 |
| block 0 | 0.001350 | 0.000010 | 0.000050 | 0.000052 |
| block 9 | 0.148529 | 0.007462 | 0.009837 | 0.000406 |
| block 19 | 0.279396 | 0.018921 | 0.025125 | 0.001018 |
| block 23 | 0.801903 | 0.037656 | 0.050135 | 0.002002 |
| block 27 | 1.900269 | 0.075833 | 0.102217 | 0.006261 |
| final norm | 0.394117 | 0.013806 | 0.020031 | 0.008412 |
| complete logits | 0.153016 | 0.028928 | 0.034059 | 0.013777 |

max-abs不是每层严格增加，但mean、RMS和relative-L2总体累积。block 26到27出现最大的max-abs跳变。

## 3. 这些数字说明什么

- 差异不是token、row或Cache copy引入，因为embedding和B2重复行全部exact；
- block 0已经出现差异，所以要继续拆第一层的Attention/FFN，而不是先改最后的argmax；
- 最终logits max-abs 0.153低于仓库已有官方BF16 0.2门，但低margin token仍可能翻转；
- “误差在容差内”与“greedy token必然相同”是两件不同的事；
- 3个fresh pair逐字段相同，现象不是随机漂移。

## 4. 下一步

后续[block0子阶段实验](block0-drift.zh-CN.md)已经完成：Attention、RoPE、residual和FFN norm
全部exact，第一处差异只在BF16 FFN output。下一步继续拆cast、gate/up、SwiGLU和down。

![Prefill layer drift](../optimization-log/assets/prefill-layer-drift.svg)

完整记录见[Experiment 106](../optimization-log/experiments/106-prefill-layer-drift.md)。
