# Step 126 — Scoped prefill Attention model counterfactual

Status: planned

Experiment 309的best exact index是QK=304681、P×V=295716，但operator admission均失败。下一步不是
默认优化，而是一次可删除的完整模型反驳。

先增加两个严格scope：`PrefillAttentionQk`与`PrefillAttentionPv`。CLI只为真实B1/2/4/8 descriptor
注册version-local index；projection、O、训练、decode和其他shape不得命中。

固定DeepSeek T2048、Q projection=296100、K/V projection=292135，跑default与QK+PV candidate：

- B1/2/4/8、两个fresh process；
- block-0 score/probability/P×V与BF16 cache；
- 完整151,936 logits Max/RMS、top1；
- prefill wall/tokens/s、peak、allocation与registry hit/miss；
- default/candidate各自within-batch行一致性。

若完整logits没有明显改善，或任一batch端到端回退>5%，删除模型/CLI路由。即使通过，也只能作为
当前gfx942/ROCm/hipBLASLt的显式策略，不能硬编码通用默认。
