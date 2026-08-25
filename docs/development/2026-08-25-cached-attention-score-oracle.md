# 2026-08-25 — cached Attention score oracle

## 用一个简单比喻理解

Attention像查资料：query是问题，cache里的每个key是一张卡片。模型先给每张卡片打分，再把分数
变成概率，最后按概率读取value。

```text
问题·卡片 → score
score → softmax probability
probability×value → context
```

以前API只给最后的context。如果新Kernel结果不对，我们不知道是哪一步错了。现在新增：

```cpp
cached_gqa_attention_scores(query, key_cache, repeats, scale)
```

它只返回第一步的所有score，shape是`[batch, query_heads, 1, cached_tokens]`。它不改cache，也
不参与默认生成。

## 为什么不是重新写一份GPU数学

HIP实现复用仓库已有的readable `cached_attention_scores_kernel`。T≤4096的默认生成仍走
`cached_attention_fused_kernel`。这样diagnostic不会偷偷改变被测路径，未来候选则必须同时对齐：

1. score；
2. softmax后概率；
3. 最终context和模型logits。

## 覆盖范围

- CPU手算6个score；
- PyTorch独立矩阵乘和GQA repeat；
- DeepSeek H12/KV2/D128；
- FP32与BF16 cache；
- B1/B2；
- T31/32/33、511/512/513、2048；
- 完整输出，不抽样；
- Max/RMS、finite、shape、连续布局、输入地址和零payload transfer；
- 错误repeats、scale、shape、stride可见。

## 本节点验证

- CPU 371/371；
- ASan/UBSan 369/369；
- 单卡HIP 191/191；
- PyTorch OperatorParity CTest通过；
- coverage audit：176个Tensor算子、42个graph API、126个测试文件。

下一步仍是测量，不是优化声明：用当前干净revision重跑DeepSeek T2048/B2/N64和rocprof时间线，
确认现在最大的阶段和Kernel。
