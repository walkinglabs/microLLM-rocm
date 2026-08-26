# Prefill Attention 核心分解诊断

## 先用简单的话说问题

Attention 可以想成翻笔记：

1. Q 和 K 做乘法，给每一页打一个“相关分数”；
2. causal softmax 把分数变成总和为 1 的权重，同时不许偷看未来；
3. P×V 按这些权重把真正的内容取回来。

上一节点只知道“取回来的内容”开始出现跨 batch 差异，却不知道前三步中的哪一步最先不同。
如果直接修改 Kernel，就像只知道汤变咸了，却先换锅而不检查盐在哪一步加入。

## 本节点只增加什么

新增 `causal_gqa_attention_diagnostics`。它返回四个明确边界：

```text
scaled_query → QK scores → probabilities → P×V output
```

模型 trace 只有在以下条件同时满足时才走这条路：

- 当前是 cached prefill；
- `capture_values=true`；
- value filter 明确点名上述四个边界之一。

没有 filter、只要 metadata，或者普通训练/推理时，模型继续走原生产路径。诊断路径会额外保存
一个 T×T score Tensor，并使用 out-of-place softmax，所以它只用于找数值差异，不能用于速度结论。

## 容易写错的地方

- 如果每次 trace 都分解 Attention，profiling 本身就会改变速度和显存；
- 如果 diagnostics 使用另一套 QK 或 P×V 实现，它观察的就不是生产数学；
- 如果只比较 shape，不比较最终输出，out-of-place softmax 可能悄悄改变结果；
- 如果 `capture_values=false` 仍触发分解，metadata-only trace 也会产生巨大临时 Tensor。

因此测试不仅检查四个 shape，还检查 CPU 每个阶段、HIP 长序列最终输出、缓存内容以及默认路径
没有新增记录。

## 实测证据

| Gate | 结果 |
|---|---:|
| CPU Debug | 376/376 |
| ASan/UBSan | 374/374 |
| PyTorch-enabled CPU | 379/379 |
| MI300X/gfx942 HIP | 195/195 |
| RCCL | 53/53 |

HIP T256 的 diagnostics 输出与生产路径逐元素完全相同。CPU tiny 模型中，点名 scores 的 trace
和无 trace 输出在 `1e-6` 内一致，BF16 K/V cache 完全相同。`capture_values=false` 即使保留同一
filter，也不会产生四个诊断记录。

![Attention core diagnostic paths](../../benchmarks/results/2026-08-26-prefill-attention-core-diagnostics/diagnostics.svg)

## 仍然没有证明什么

这个节点只证明“显微镜不会默认开、开了以后边界可信”。它还没有证明 DeepSeek T2048 的第一处
差异在 QK、softmax 或 P×V。下一节点必须用 B1 参考和 B2/B4/B8 第一行做完整阶段比较，并且
避免把几百 MB 的 score JSON 永久留在仓库里。
