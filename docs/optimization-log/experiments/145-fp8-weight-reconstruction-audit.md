# Experiment 145：权重本身只改善约1%，模型却能正负波动几十倍

## 为什么需要这一步

Exp143的全模型结果方向相反：DeepSeek变准、Qwen变差。为了不盲猜范围，我们在相同真实权重上
比较scalar和per-output-channel的FP8重建误差，再按Linear家族合并。

| 模型/分组 | scalar rel-L2 | column rel-L2 | column/scalar |
|---|---:|---:|---:|
| Qwen 全部 | 0.026491 | 0.026299 | 0.992762 |
| Qwen Attention | 0.026472 | 0.026195 | **0.989549** |
| Qwen FFN | 0.026494 | 0.026319 | 0.993372 |
| Deep 全部 | 0.026493 | 0.026386 | 0.995969 |
| Deep Attention | 0.026491 | 0.026351 | 0.994741 |
| Deep FFN | 0.026497 | 0.026413 | 0.996851 |
| Deep output head | 0.026465 | 0.026209 | **0.990329** |

![Weight reconstruction audit](../assets/fp8-weight-reconstruction-audit.svg)

365个Tensor全部合法；Qwen 166个、Deep 189个逐Tensor改善，但全模型合并只改善0.72%/0.40%。
少数down projection还轻微变差约0.06%/0.13%。即使scale max/median最高达到18.3×，E4M3相对
精度使per-column并没有自动带来巨大重建收益。

## 和Exp143放在一起看

权重重建变化不到1%，DeepSeek模型RMS却改善33%–59%，Qwen反而恶化约28%。这不是矛盾：
Transformer连续传播、残差抵消和原生GEMM会改变误差方向，小局部差异可以被放大或抵消。

因此外部权重审计只能选“下一次改哪里”，不能预测最终模型误差。

## 下一步范围

Deep output head是最佳分组，而且只有一个Linear；Qwen tied embeddings没有独立output head。
所以最小反事实是`output-head-only`：

- Qwen应与device-Tensor baseline完全相同；
- Deep只给LM head逐列scale，其他196个Linear保持scalar；
- 每次forward最多增加一个post-scale，而不是197个；
- 必须回到microLLM native T8/T512完整logits和三进程性能门。

Attention范围仍保留为下一候选，但不与output head同时改变。
