# Positions-aware decode：不同页数的请求终于一起算

## 1. 为什么active compaction还不够

上一节点已经删掉空slot，但两个真实请求位置不同时仍逐row运行完整模型：

```text
row 0 position=3 → B1 Transformer
row 1 position=6 → B1 Transformer
```

这样QKV、FFN和output head都失去batch并行。

## 2. 新路径传入两张小表

```text
positions  = [3, 6]   每个active请求当前写到第几页
cache_rows = [0, 1]   它在共享KV Storage里的物理row
```

模型先对A个active请求一起做Embedding、QKV、FFN和output head。只有三个需要知道页数的算子读取
上面的小表：

1. RoPE按`positions[a]`算每行旋转角度；
2. K/V store写入`cache_rows[a], positions[a]`；
3. cached Attention只读取该row的`0..positions[a]`。

这条路径同时支持普通成对RoPE、Qwen split-half RoPE、Q/K bias、FP32/BF16 Cache和大于4096的
score/softmax/context fallback。

## 3. 哪些东西不能改变

- inactive row完整capacity不变；
- active row只推进1；
- 每row logits等于独立B1；
- Cache地址、allocated/active bytes和slot生命周期不变；
- full+uniform仍走旧batch fast path；
- CPU和HIP必须对齐，HIP执行区间不能偷偷D2H payload。

## 4. 性能证据

五个Release divergent shape中四个单轮提高18%–56%，一个R8/S2单轮下降14.6%。没有把它藏掉，
而是冻结上一提交做三对交替A/B。最终中位数：

- R8/S2：1.295×；
- R8/S4：1.670×；
- R4/S4：1.610×。

9/9逐对candidate更快，输出checksum完全相同。R8/S2单轮回退因此被反驳，但仍保留在原始矩阵。

## 5. 下一步

现在不同position的真实row已经并行，但每step仍从CPU创建`positions`和`cache_rows`并H2D，prompt
prefill也逐row执行。下一步应先profile新时间线，再选择持久device metadata或batched prefill，
不能猜哪个更重要。

这一步已经在[continuous-only profile](continuous-profile.zh-CN.md)完成；trace随后否定了GPU logits
scatter候选，因此positions-aware主路径保持原实现。

详细数据见 [Experiment 098](../optimization-log/experiments/098-positions-aware-decode.md)。
