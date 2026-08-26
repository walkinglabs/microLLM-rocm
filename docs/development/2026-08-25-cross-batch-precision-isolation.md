# 哪一部分低精度计算放大了Batch差异

## 保持其他条件不变

如果同时更换KV Cache、Attention算法和权重精度，就不知道哪个变化造成结果。这里固定BF16 KV和
同一个保序Attention，只开关两组权重：FFN与Attention。

## 结果像四级放大器

全FP32 Linear仍有约0.00135差异，这是GPU针对不同batch shape选择不同GEMM算法产生的底噪。
只把Attention权重变BF16，最大差异约0.02097；只把FFN变BF16，达到0.06299；当前两者都BF16是
0.06757。

因此FFN是主要放大器，Attention是次要贡献。这里不能说“FFN有Bug”：BF16本来就会圆整，问题是
哪一层的batch-shape差异被放大到会改变后续token。

## 下一步只看每个Block出口

模型有28层。我们先保存每层输出，而不是立刻保存所有中间Tensor：

```text
embedding → block0 → block1 → ... → block27 → final norm → logits
```

找到误差第一次明显跳大的层后，再只打开那一层的FFN norm、gate、up、激活和down细节。这样trace
文件更小，因果也更清楚。
