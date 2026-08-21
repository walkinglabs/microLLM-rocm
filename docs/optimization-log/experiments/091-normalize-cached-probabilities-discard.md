# Experiment 091 — 除法只做一次，为什么没有更快？

cached fused Attention先把`exp(score-max)`放进shared memory。旧Value阶段对每个输出column都写：

```text
shared_score[position] / denominator × value[position,column]
```

候选先并行归一化每个position，再让所有column直接乘已经归一化的概率。Key dot、exp、规约顺序、
Value累加顺序和block布局不变；代价是一轮shared读写和一次barrier。

## 数值门

4个focused HIP tests通过。DeepSeek T2048：

| Shape | logits | max-abs | RMSE | token |
|---|---:|---:|---:|---|
| B1 | 151,936 | 0 | 0 | 相同 |
| B8 | 1,215,488 | 0 | 0 | 相同 |

两份binary输出逐byte相同，所以这个候选有资格计时。

## 三对交替Release性能

| Batch | baseline median | candidate median | 速度比 |
|---:|---:|---:|---:|
| 1 | 67.30 tok/s | 66.90 tok/s | 0.994× |
| 8 | 513.16 tok/s | 511.84 tok/s | 0.997× |

allocator固定94次backend allocation，D2H固定3次，peak/token完全相同。负差异不足5%，但也没有
任何稳定收益。最可能的解释是编译器已经把不变量division处理得足够好，而新增shared pass与barrier
抵消甚至超过节省。

![Normalize cached probabilities discard](../assets/normalize-cached-probabilities-discard.svg)

## 决定

候选完整回退。位级正确不是保留理由；没有速度或内存收益的额外同步只会增加维护成本。

下一轮若继续Value阶段，应改变读取粒度而保持每个column的position累加顺序，例如让一个线程处理
两个相邻BF16 Value column；仍必须先过百万完整logits门。

数据见[`091-data`](091-data/)。
