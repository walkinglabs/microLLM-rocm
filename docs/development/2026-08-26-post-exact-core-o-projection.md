# 找到O Projection

Attention core像“查资料”，O projection像“把查到的内容重新写回模型的语言”。我们让查资料结果在
不同batch完全一样，然后发现第一次新差异发生在重新写回这一步。

B2/B4/B8的context都是exact；O之后最大差异约2.77e-5到3.34e-5，并且同batch相同输入也分叉。
因此下一步只修O，不把FFN一起改。

![O projection boundary](../../benchmarks/results/2026-08-26-post-exact-core-block0-trace/post-exact-core-trace.svg)
