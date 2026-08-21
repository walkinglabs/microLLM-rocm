# Experiment 092 — 两个Value column一起算，为什么反而更慢？

Key pair会改变同一个dot的累加代码形态，Experiment 088/089因此失败。Value阶段不同：每个输出
column本来就有独立accumulator。候选让一个线程维护相邻两个column，两个total仍各自按position
0→T顺序累加。

## 候选边界

- Key dot、softmax和shared score完全不变；
- BF16 Value一次32-bit读取，显式恢复两个公开scalar；
- 每个position的probability只读取/计算一次，分别乘两个Value；
- first/second total完全独立；
- odd width和FP32走原scalar路径。

## 完整数值门

DeepSeek T2048 B1/B8的151,936/1,215,488个cached logits全部位级一致，max/RMSE为0，token
相同。它证明“保持每个column的position顺序”足以守住当前官方轨迹。

## 三对交替性能

| Batch | baseline median | candidate median | 速度比 |
|---:|---:|---:|---:|
| 1 | 67.28 tok/s | 66.47 tok/s | 0.988× |
| 8 | 513.72 tok/s | 508.14 tok/s | 0.989× |

两项都是稳定约1%的回退。allocator固定94，D2H固定3，peak与token相同。可能原因是Value原访问已经
按column连续合并；pair写法把活跃线程从width降到width/2，并增加拆包，收益不足以抵消并行度下降。

![BF16 paired Value load discard](../assets/bf16-paired-value-load-discard.svg)

## 决定与局部搜索边界

候选完整回退。结合Experiment 088–092：

- Key pair：官方精度失败；
- shared probability预归一化：位级正确但0.994×/0.997×；
- Value pair：位级正确但0.988×/0.989×；
- 旧64/128-thread与query shared staging也已有失败证据。

继续排列标量读法已经没有新信息。下一版cached Attention必须改变更大的算法边界，例如wave协作
dot、online softmax或可验证的MFMA tile，并先建立逐position score中间门。

数据见[`092-data`](092-data/)。
