# Step 130 — Scoped O projection complete model gate

Status: planned

对照`exact-core`与`exact-core+O296100`，固定DeepSeek T2048、B1/2/4/8、两个fresh process：

- BF16 cache和完整151,936 logits；
- O output trace因果证据复用Experiment 313；
- full prefill wall/tokens/s、peak、allocation、registry 5/168；
- Max/RMS都至少改善10%，每个batch≥0.95×。

失败则O默认拒绝但scope保留为诊断工具；无论结果如何，下一诊断细分FFN gate/up/activation/down。
