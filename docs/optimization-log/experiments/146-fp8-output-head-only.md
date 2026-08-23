# Experiment 146：只改LM head，最终logits完全没变

## 最先犯的基线错误

最初计划用Exp129/135比较，但Exp135是host Tensor-amax；output-head-only未选中的196个Linear
实际使用device Tensor-amax。两个不同量化实现不能拿来做单变量结论。

所以我们没有接受看似改善的数据，而是在同一个`cdcb8e7` binary上追加完整control。两套各
36个worker，总计72个worker、24个FP8目标行。

## 同revision结果

| 模型/上下文 | Max变化 | RMS变化 | TPS变化 | 额外常驻 |
|---|---:|---:|---:|---:|
| Qwen T8 | 0 | 0 | -4.52%（不作正式门） | 0 |
| Qwen T512 | 0 | 0 | -0.85% | 0 |
| Deep T8 | 0 | 0 | -0.06%（不作正式门） | +607,740 B |
| Deep T512 | 0 | 0 | -0.52% | +607,740 B |

![Output-head-only counterfactual](../assets/fp8-output-head-only.svg)

Qwen tied head正确地执行0次post-scale；Deep每个worker执行4次，也就是一个head×4次forward。
但Deep Max/RMS与control逐值完全相同。速度门过了，数值收益条件没过，因此targeted keep=false。
完整FP8精度门仍0/4通过。

这里的“Max/RMS相同”不证明两个151,936维向量逐元素bit-exact，但已经足够推翻预先写下的
“Deep Max和RMS都改善”保留条件；没有必要为零聚合收益增加永久API。

## 决定

删除被拒绝的output-head-only公共scope，保留Exp146复现和审查记录。下一候选可以研究Attention
范围，因为Exp145中它是两个模型共同最好的多Tensor分组；但必须新建单变量策略，不能在这个
失败scope上继续叠选项。
