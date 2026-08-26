# Step 123 — First drift after exact block-0 cache

Status: complete; Attention context selected

Experiment 305让Block 0 BF16 K/V在B1/B2/B4/B8完全位级相同，但完整logits仍随batch漂移。这个
干净反事实允许我们从cache之后继续定位。

固定DeepSeek T2048、Q=296100、K/V=292135、BF16 KV、B1/B2/B4/B8、两个fresh process，记录
Block 0：

- cache_key/cache_value（必须继续exact）；
- causal Attention context；
- Attention output；
- residual与FFN norm；
- FP32 gate/up/activated/down；
- block output。

详细Tensor仍只捕获前两个batch row，不做性能声明。若context首次漂移，审计full-prefill Attention
core；若context exact而output projection先漂移，进入同形FP32 O solution审计；若FFN先漂移，进入
大M gate/up/down FP32 solution审计。候选仍不进入默认。

结果：Cache及此前边界全exact；第一处新差异是Attention context，B4/B8 Max为`0.00033218`，同
batch context行仍exact。O projection后within-batch行开始不同。Step 124拆QK/softmax/PV。
