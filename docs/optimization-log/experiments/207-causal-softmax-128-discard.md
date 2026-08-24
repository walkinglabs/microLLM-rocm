# Experiment 207 — 128-thread causal softmax 被算子门拒绝

Status: explicit primitive retained; model/CLI policy removed

## 假设

当前register-cached causal softmax每行使用256线程。T512时每线程最多处理2个可见值，但两次
block reduction都跨4个wave。候选改用128线程，让每线程最多保存4个值，期望用更少wave和
同步换取更低延迟。T1024最多8值，仍不超过现有局部数组；T2048会达到16值，因此明确fallback。

为了让128线程正确工作，block reduction的stride从编译期256改为运行时`blockDim.x`。默认
256线程的归约树完全不变。

## 正确性

HIP直接覆盖T256/512/1024，逐项对CPU reference；最大误差`1.86e-9`、RMS不超过
`6.65e-11`。T2048继续走默认256线程边界测试。公共默认`causal_softmax()`不变；候选只通过
`causal_softmax_with_implementation(..., Rows128)`显式调用。

## 三进程算子矩阵

![128-thread causal softmax discard](../assets/causal-softmax-128-discard.svg)

每个case/策略三个新进程，每进程3次热身、20次Event测量：

| Family | T256 | T512 | T1024 |
|---|---:|---:|---:|
| Qwen heads=14 | 1.0168× | 1.0255× | 1.0127× |
| DeepSeek heads=12 | 1.0063× | 1.0071× | 1.0214× |

只有4/6通过1.01门，关键的DeepSeek T512只有1.0071×。单次pilot曾看到约1.022×，三进程结果
推翻了它。

## 决策

停止整模实验并删除模型/CLI开关。算子都没有跨模型通过时，端到端更不可能形成可靠收益。
保留显式低层primitive、完整数值测试和benchmark，方便其他GPU或shape重新研究；Auto和所有
模型路径仍为256线程。

这也关闭了当前“只调block size”的softmax方向。若继续优化，需要改变更大的结构，例如
online Attention避免物化概率，而不是再做64/128/256的小范围线程扫描。

原始证据：[operator matrix](../../../benchmarks/results/2026-08-24-causal-softmax-128-operator/)。
