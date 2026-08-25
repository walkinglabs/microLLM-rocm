# 2026-08-25 — 现在每层只剩一进一出两次转换

两个RMSNorm都直接写BF16工作区后，重新profile看到每层只剩两次精度转换：一次
从FP32进BF16，一次从BF16回FP32。

![Post Attention Norm profile](../optimization-log/assets/post-bf16-attention-norm-profile.svg)

下一步不先猜怎么融合，而是先确定这两次分别属于哪个模型边界。
