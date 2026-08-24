# Experiment 218 — 合成一张大卷子也没有更快

Status: `discard before Autograd route`

## 为什么还要测

Experiment 217 说明当前库没有可用的 FP32 GroupedGemm。另一条数学等价路线是把 Q/K/V 或
gate/up 的 output gradient 按列拼在一起，再做一次普通大 GEMM：

```text
[dQ | dK | dV]
       ↓
inputᵀ @ packed_gradient → packed_weight_gradient
```

packed output 可以通过 Tensor view 分给三个参数，因此不需要额外 split copy。但 output gradient
每一步都在不同 Storage 中，pack 本身不能免费。

## 公平计时

候选每次计时包含2/3次真实 `hipMemcpy2DAsync` D2D pack和一次大 GEMM。baseline是当前2/3次
独立 hipBLASLt GEMM。完整结果逐元素比较。

| Model | Projection | Median speedup | Max error |
|---|---|---:|---:|
| Qwen | QKV | 0.979× | 1.15e-7 |
| Qwen | gate/up | 0.835× | 0 |
| DeepSeek | QKV | 0.897× | 9.69e-8 |
| DeepSeek | gate/up | 0.931× | 4.19e-8 |

![Packed weight-gradient discarded](../assets/packed-weight-gradient-discard.svg)

## 决定

0/4 case 通过1.05算子门，所以不增加 pack/split Autograd 节点，也不增加最高105 MiB的packed
output。Grouped与packed两条weight-gradient组合路线均关闭。

下一方向必须针对真正耗时的单个 GEMM family 做 exact-shape算法或更底层 Kernel，而不是继续
改变提交数量。

原始证据在
[`benchmarks/results/2026-08-24-packed-weight-gradient-discard/`](../../../benchmarks/results/2026-08-24-packed-weight-gradient-discard/)。
