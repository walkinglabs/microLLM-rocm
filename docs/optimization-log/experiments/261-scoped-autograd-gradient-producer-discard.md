# Experiment 261 — 算子快很多，接进Autograd为什么没有收益

Status: `discard and remove Autograd route`

在已构建graph上重复backward，使用与operator门相同四个Model-S shape和tiny反例；每shape三个
fresh process、5 warm-up、40次测量，轮换baseline/direct和shape顺序。

| Shape | Event | Wall | 结论 |
|---|---:|---:|---|
| head T32 | 0.993× | 0.994× | reject |
| FFN T32 | 0.995× | 0.999× | reject |
| Attention T32 | 0.976× | 0.991× | reject |
| head T512 | 1.035× | 1.018× | reject |
| tiny | 1.001× | 1.005× | reject |

![Scoped Autograd producer discard](../assets/scoped-autograd-gradient-producer-discard.svg)

15个完整gradient exact、地址保持，每次logical allocation少1，但0/5同时过Event/Wall 1.05门。
原因是普通Autograd首次leaf contribution本来就直接接管producer Tensor，没有leaf add；candidate
只把缓存allocation换成target状态管理，GPU工作几乎相同。

撤回scoped dispatch、overwrite/zero target和benchmark runner；保留独立caller-owned operator。
下一多卡节点转向gradient-ready顺序与真实通信/计算重叠，不再搜索leaf target微优化。

证据：[`Autograd producer matrix`](../../../benchmarks/results/2026-08-25-autograd-gradient-producer-matrix/)
