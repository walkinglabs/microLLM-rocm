# Experiment 314 — O更准确了，但B1更慢了

## 先说结论

给cached-prefill的O投影固定使用296100后，完整logits的全局Max误差从`0.001562`降到
`0.001175`，RMS从`0.000310`降到`0.000209`，分别改善24.7%和32.6%。但是B1 prefill只有
原来的0.944×，低于事先写下的0.95门。因此这个方案被拒绝，不进入默认路径。

## 像初中生一样理解它

可以把一层模型想成一条接力赛。前面的Attention core已经换成“每次用同一种顺序算”的选手，O投影
是下一棒。把O也固定以后，传到后面的数字确实更整齐了，但B1跑得明显更慢。一次改动要同时满足
“答案更稳”和“不能明显变慢”，只满足前一半不能通过。

## 怎么测

- 模型：DeepSeek-R1-Distill-Qwen-1.5B；
- 输入长度：2048；batch：1、2、4、8；
- baseline：Q/K/V和QK/P×V使用exact诊断index；
- candidate：在baseline之上只给O投影增加296100；
- precision：16个独立进程；performance：16个反向排序独立进程；
- 比较完整151,936 logits、BF16 cache、prefill、峰值显存和后端分配。

| Batch | Candidate prefill | Baseline Max | Candidate Max | 结果 |
|---:|---:|---:|---:|---|
| 1 | 0.944× | 0 | 0 | 性能失败 |
| 2 | 0.996× | 0.001562 | 0.001175 | 数值改善 |
| 4 | 0.991× | 0.001204 | 0.000576 | 数值改善 |
| 8 | 1.001× | 0.000823 | 0.001134 | Max恶化 |

峰值显存和后端分配没有变化。原始证据、汇总和图在
[`benchmarks/results/2026-08-26-fp32-prefill-o-model-gate`](../../../benchmarks/results/2026-08-26-fp32-prefill-o-model-gate/README.md)。

## 没有声称什么

- 没有把O=296100设成默认；
- 没有用全局RMS改善掩盖B1性能失败和B8 Max恶化；
- 没有把相对已拒绝exact-core的改善写成相对真实upstream的改善。

下一步只做一次真实upstream对batch-selective exact stack的完整反驳；如果仍不过门，就展开FFN内部trace。
