# Experiment 315 — 最后一次exact组合仍然没有通过

## 结论

相对真实upstream，固定的batch-selective exact stack在Release下没有明显变慢：B1/2/4/8分别为
0.997×、0.987×、1.020×、1.008×。但是完整logit Max从`0.001253`升到`0.001340`，恶化6.9%；
RMS从`0.000291`降到`0.000284`，只改善2.5%。Max和RMS都必须至少改善10%，所以候选被拒绝。

## 为什么这是最后一次组合

前面的实验分别证明Q/K/V、QK/P×V和O可以让某个局部边界更整齐。但模型像很长的接力赛：改变一棒
的加法顺序，会改变后面许多棒收到的最后几位数字。按batch拼出局部看起来最好的方案，并没有得到
稳定的完整模型改善。继续从旧表格挑更多组合会变成事后挑答案，而不是可反驳实验。

## 固定策略和证据

- B1：upstream；
- B2/B4：QK=304681、P×V=295716、O=296100；
- B8：QK=304681、P×V=295716、O关闭；
- 16个独立precision进程、16个反向排序performance进程；
- Release构建；完整BF16 cache和151,936 logits；
- 每个进程校验solution注册、缓存和dispatch计数。

| Batch | Prefill比值 | Upstream Max | Candidate Max |
|---:|---:|---:|---:|
| 1 | 0.997× | 0 | 0 |
| 2 | 0.987× | 0.000747 | 0.001340 |
| 4 | 1.020× | 0.001253 | 0.001217 |
| 8 | 1.008× | 0.001227 | 0.001212 |

峰值显存和后端分配逐batch相同。原始结果与图见
[`benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate`](../../../benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate/README.md)。

## 下一步

停止组合exact Linear solution。保留scope作显微镜，把Block-0 FFN拆成gate、up、SwiGLU activation和
down output，找到聚合FFN output之前真正的第一处差异。
