# Experiment 313：O Exact以后，第一处差异移动到FFN Output

Status: FFN selected; O model gate still required

在exact Q/K/V/QK/P×V控制上加入O=296100。8个fresh process、17个block-0边界显示：context、O output、
residual、FFN norm在B2/B4/B8跨batch和同batch全部bitwise。`ffn_output`统一首次非零，Max依次为
`2.193e-5/1.431e-5/1.812e-5`。

![Post exact O](../../../benchmarks/results/2026-08-26-post-exact-o-block0-trace/post-exact-o-trace.svg)

这证明O scope局部因果成立，但没有证明O策略可保留。下一节点先跑exact-core vs exact-core+O的完整
logits/prefill/peak；之后再让prefill FFN暴露gate/up/activated/down，定位内部第一处。

证据：[`result directory`](../../../benchmarks/results/2026-08-26-post-exact-o-block0-trace/)
