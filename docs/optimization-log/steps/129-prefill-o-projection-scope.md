# Step 129 — Scoped prefill O projection counterfactual

Status: trace gate passed; model route rejected by Experiment 314

新增`PrefillAttentionOutputProjection`，只让cached-prefill的Attention O Linear使用。Q/K/V projection、
QK/P×V、FFN、训练和decode保持独立。

候选先使用已通过相同M/K/N row-invariance的296100。固定exact upstream/core控制，检查：

- O output跨/内B1/2/4/8是否bitwise；
- registry 5 entries、每层6次hit，共168 dispatch/forward；
- 完整logits Max/RMS、top1；
- full prefill、peak、allocation。

若O exact但完整logits无稳健改善，或任一batch<0.95×，拒绝；再用trace判断FFN是否首差。

Experiment 313证明O与FFN norm跨/内batchexact，FFN output首差。Experiment 314的完整模型门让
Max/RMS改善24.7%/32.6%，但B1 prefill只有0.944×，因此scope保留为诊断工具，不进入默认路径。
