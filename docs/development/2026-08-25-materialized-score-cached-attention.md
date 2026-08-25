# 2026-08-25：并行算Score，但不改变Softmax和P·V顺序

## 上一个候选为什么失败

split-sequence把每段分别做max、sum和P·V，再用log-sum-exp合并。数学等价，但浮点加法顺序改变；
DeepSeek经过28层和64步后，完整logits漂移到Max/RMS 0.05691/0.01370。

新候选只保留已经证明有效的部分：让很多blocks并行计算每个position的Q·K score。

## 两个Kernel

```text
Kernel 1：每个线程负责一个 batch/head/position
          用原来完全相同的column 0→D dot顺序
          写全局FP32 scores[B,H,T]

Kernel 2：每个batch/head一个block
          按原fused相同的position→thread映射读score
          使用原block_reduce_max
          使用原exp和block_reduce_sum
          使用原column与position P·V循环
```

这样仍把Q·K grid从B×H提高到约`B×H×T/256`，但max、分母和value累加的浮点顺序不变。代价是
一块`B×H×T×4 bytes`的global score Tensor和第二次launch。DeepSeek T2048/B2为196,608 bytes。

## 当前正确性证据

公开研究接口是`cached_gqa_attention_materialized_scores`，默认模型不调用它：

- CPU小数值与当前cached Attention完全相同；
- PyTorch `softmax(QK^T*scale)@V`对齐；
- MI300X DeepSeek H12/KV2/D128覆盖B1/B2、FP32/BF16；
- T31/32/33、T511/512/513和T2048共16格；
- 16格完整context与当前fused逐元素位级相同；
- 输入地址不变，计算区间0 payload transfer；
- 非法shape/stride和T>4096明确拒绝。

下一节点分别测current和materialized的Event/wall、2次逻辑allocation、热backend allocation和score
bytes。至少1.05x才进入模型；位级精度只是准入条件，不是保留理由。

## 保持数学顺序的线程映射接口

后续profile发现第二个Kernel成为最大单项。研究重载增加一个显式`finalize_threads`参数，只接受
64、128或256；不传参数仍是原来的256。64/128不是简单改变block reduction的宽度，而是用较少
物理线程模拟同样的256个逻辑lane：

```text
逻辑lane 0仍累加position 0,256,512,...
逻辑lane 1仍累加position 1,257,513,...
...
共享内存仍按128,64,...,1的相同树归约
每个输出column仍按position 0→T累加
```

因此这个接口只回答“少一些物理线程是否更适合调度”，不把数值顺序变化混进同一个实验。
CPU、小shape PyTorch oracle和16格MI300X长短边界都覆盖默认/64/128；非法线程数明确失败。
模型Auto路由不读取这个参数。正式性能结论必须来自
`cached_attention_finalize_mapping_matrix.py`，不能从一次手工计时得出。

## 只拆P×V的精度隔离接口

`cached_gqa_attention_split_pv_exact_softmax`继续使用相同的并行score和256-lane max/denominator树。
它把归一化概率写到FP32 Tensor，只将最后的`probability × value`序列分成连续片段并按split顺序
合并。

S1不是性能候选，而是隔离证明：它有额外probability/partial Tensor和launch，但必须与当前
materialized context位级相同。只有S1通过后，S2/4/8/16的差异才可以解释为P×V累加树变化。
CPU、PyTorch、小shape以及MI300X T31–2048、B1/B2、FP32/BF16均覆盖接口和非法split。
