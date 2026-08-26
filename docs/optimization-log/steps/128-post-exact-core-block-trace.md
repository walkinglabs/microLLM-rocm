# Step 128 — First drift after an exact Attention core

Status: completed by Experiment 312

QK/P×V solution优化线已关闭，但Experiment 310提供了一个有用的诊断控制：block-0 core与cache跨batch
exact。使用同一Q/K/V/QK/P×V index，不做性能声明，继续记录：

```text
P×V output → context layout → O projection → residual → FFN norm
→ gate/up → SwiGLU → down → block output
```

固定DeepSeek T2048、B1/2/4/8、两个fresh process，完整比较前两个相同输入行。若O projection首差，
下一实验只做O的真实M2048/4096/8192/16384 row-invariance与scope；若FFN首差，转到相应Linear。

exact-core indices只用于显微镜，不重新参加默认性能门。

结果：B2/B4/B8 context跨/内batch全exact，`attention.output`统一首差。详见
[`Experiment 312`](../experiments/312-post-exact-core-o-projection.md)。
