# Experiment 312：Core之后第一处新差异是O Projection

Status: O projection selected

使用Q/K/V/QK/P×V exact indices作为诊断控制，不做性能声明。B1/2/4/8、两个fresh process、17个
block-0边界完整比较。

三组非B1 case的`attention.context`跨batch和同batch全部位级相同。下一边界`attention.output`统一
首次非零：B2/B4 Max `3.3379e-5`，B8 `2.7657e-5`。同batch两条相同输入行也从O projection开始不同。

后续residual、FFN norm/output只传播或改变这组差异，不是第一来源。

![Post exact core trace](../../../benchmarks/results/2026-08-26-post-exact-core-block0-trace/post-exact-core-trace.svg)

## 决定

下一节点只增加`PrefillAttentionOutputProjection` scope。O与Q projection同为M×1536乘1536×1536，
Experiment 304已证明296100跨M2048/4096/8192/16384位级一致。先做scope/route门，再检查O输出、
完整logits与端到端；不重新开启QK/P×V路线。

证据：[`result directory`](../../../benchmarks/results/2026-08-26-post-exact-core-block0-trace/)
