# Step 129 — Scoped prefill O projection counterfactual

Status: trace gate passed; model gate pending

新增`PrefillAttentionOutputProjection`，只让cached-prefill的Attention O Linear使用。Q/K/V projection、
QK/P×V、FFN、训练和decode保持独立。

候选先使用已通过相同M/K/N row-invariance的296100。固定exact upstream/core控制，检查：

- O output跨/内B1/2/4/8是否bitwise；
- registry 5 entries、每层6次hit，共168 dispatch/forward；
- 完整logits Max/RMS、top1；
- full prefill、peak、allocation。

若O exact但完整logits无稳健改善，或任一batch<0.95×，拒绝；再用trace判断FFN是否首差。

Experiment 313证明O与FFN norm跨/内batchexact，FFN output首差。完整O模型门仍必须执行。
