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
