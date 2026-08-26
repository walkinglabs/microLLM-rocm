# Experiment 298：第一处低精度放大是FFN输入Cast

Status: block-0-only FP32 counterfactual selected

## 把第0个Block打开

DeepSeek T2048 cached step0固定同一prompt，比较B1和B2第0行。FP32 Linear与BF16 FFN-only各跑
两个fresh process，记录Block 0内Q/K/V、RoPE、Attention、残差、FFN和完整输出。

![Cached block-0 detail](../../../benchmarks/results/2026-08-26-deepseek-cached-block-detail/block-detail.svg)

| 边界 | Max | RMS | relative-L2 | 相邻解释 |
|---|---:|---:|---:|---|
| Attention context | 5.62e-5 | 1.53e-6 | 5.74e-6 | 第一处通用batch漂移 |
| FFN norm | 2.98e-6 | 7.22e-7 | 4.32e-6 | 仍是FP32底噪 |
| BF16 input | 4.88e-4 | 1.69e-5 | 1.01e-4 | Max/rel-L2放大163.84x/23.38x |
| BF16 gate | 0.0078125 | 1.998e-4 | 4.31e-4 | 第一处Max超过1e-3 |
| BF16 up | 0.00390625 | 1.273e-4 | 4.22e-4 | 同一输入的第二分支 |
| BF16 down | 0.00390625 | 3.481e-4 | 0.001143 | 本层relative-L2峰值 |
| Block output | 0.003909 | 3.482e-4 | 4.78e-4 | 与Experiment 297闭合 |

Attention norm、Q/K/V投影和RoPE全部位级相同。Attention context先产生5.62e-5的小差异；它经过
残差与FFN norm后Max只剩2.98e-6。把这组FP32输入转换成BF16时，跨过不同舍入桶的元素把Max放大
163.84倍；gate再把Max放大16倍。FP32 gate只有7.75e-6，因此BF16 gate相对它约1008倍。

两个fresh run完全相同，B2两行在Block 0全部位级相同。Experiment 297中B2行间差异到Block 1才
出现，因此当前结果也排除了Block 0行索引错误。

## 决定

不把全模型退回FP32，也不先换GEMM算法。下一反驳实验只保留Block 0 FFN的FP32权重和激活，其他
27层继续BF16。若最终logits漂移显著下降，就支持“早层量化误差主导累积”；若几乎不变，则误差主要
来自每层重复注入，应该改全局cast/scale策略。默认precision和scheduler继续冻结。

证据：[`cached block detail`](../../../benchmarks/results/2026-08-26-deepseek-cached-block-detail/)
