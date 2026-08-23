# Experiment 137：Qwen在21层突变，DeepSeek在27层放大

| 模型 | 首次非零 | 最大隐藏层跳变 | 关键层rel-L2 | final logits max/RMS/rel-L2 |
|---|---|---|---:|---:|
| Qwen | block0 | block20→21 `+0.2028` | block21 `0.2121` | 0.999/0.192/0.0745 |
| Deep | block0 | block26→27 `+0.0764` | block27 `0.1150` | 1.243/0.438/0.2413 |

![FP8 layer drift](../assets/fp8-layer-drift.svg)

Qwen block2–20 relative-L2大多约0.5%–0.9%，21层突然到21.2%。Deep block1–25约2.5%，26层
3.9%，27层11.5%；final norm降低相对误差，但词表投影把Deep logits放大到24.1%。

Qwen最大max/RMS在final norm，Deep最大隐藏max/RMS在block27；不能只按relative-L2选层。下一
实验展开Qwen21/Deep27内部norm、QKV、RoPE、context、residual、FFN与activated，找出该block
内第一个大跳，其他层不增加trace复杂度。
