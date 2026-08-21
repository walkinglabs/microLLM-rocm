# Experiment 098 — 不同position的真实row开始并行

Experiment 097只消除了空row，多个真实row仍逐个执行B1模型。本节点让active batch的Embedding、
QKV、FFN和output head共同执行，并把position差异下沉到三个算子。

## 算子合同

```text
positions[A]  每个active请求写入位置
cache_rows[A] 每个active请求对应共享Cache物理row
```

- RoPE：interleaved、split-half和split-half+bias逐row角度；
- KV store：`current[A,H,1,D]`映射到`cache_rows[a],positions[a]`；
- cached GQA：每row只看`positions[a]+1`个token；
- prefix≤4096走fused block，prefix>4096走masked scores/softmax/context。

![Positions-aware decode](../assets/positions-aware-decode.svg)

## 正确性

CPU每个新算子与标量row reference对齐；HIP覆盖FP32/BF16、两种RoPE、Q/K bias、GQA和4097
fallback，执行区间0次payload D2H。完整模型active logits逐row等于独立B1，inactive完整capacity、
Storage地址、Cache字节与请求状态不变。

## Release矩阵与失败

| shape | candidate/097 | continuous/reference | 判断 |
|---|---:|---:|---|
| R2/S2 | 1.263× | 1.213× | 正向 |
| R4/S2 | 1.180× | 1.196× | 正向 |
| R8/S2 | 0.854× | 0.899× | 单轮负面，必须反驳 |
| R4/S4 | 1.460× | 1.574× | 正向 |
| R8/S4 | 1.555× | 1.617× | 正向 |

uniform控制没进入positions-aware，绝对TPS与reference/static一起下降约6%，说明机器状态变化。R8/S2
即使归一化仍负面，所以没有直接接受。

## 三shape严格交替A/B

| shape | baseline median | candidate median | speedup | normalized old→new |
|---|---:|---:|---:|---:|
| R8/S2 | 2141.39 | 2773.93 | 1.295× | 0.953→1.151 |
| R8/S4 | 2352.76 | 3929.77 | 1.670× | 0.990→1.636 |
| R4/S4 | 2343.20 | 3772.16 | 1.610× | 0.980→1.586 |

9/9逐对candidate更快，18个进程每shape checksum唯一。R8/S2单轮回退被交替实验推翻，但原始
negative row仍保留。positions-aware calls在candidate中严格等于compacted calls，uniform为0。

## 剩余问题

每step仍从host构造positions/row mapping并H2D；每个新prompt仍用临时B1 prefill；固定KV capacity
字节利用率仍低。下一节点先profile新路径，判断metadata H2D、逐row prefill还是Attention Kernel是
新主要瓶颈。

原始数据见 [`098-data`](098-data/)。
