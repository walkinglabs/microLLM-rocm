# 2026-08-25：把一个长序列分给多个GPU Block

## 旧办法为什么可能闲着很多计算单元

当前cached Attention给每个`batch × head`分配一个GPU Block。DeepSeek有12个query heads，所以
B1只启动12个blocks，B2只启动24个；当前MI300X有304个计算单元。一个block内部要从头读完
T512或T2048的历史，很多计算单元没有拿到独立工作。

这不自动证明“blocks越多越快”。拆分以后要多写临时结果，还要再启动一个合并Kernel。因此本次
只实现一个显式研究原语，不改默认推理路线。

## 两阶段算法

假设把一个head的历史切成S段：

```text
第一阶段：每段一个Block
  算本段score
  找本段最大值m
  算本段分母d = sum(exp(score - m))
  算本段加权Value n = sum(exp(score - m) * value)

第二阶段：每个head一个Block
  找所有段的总最大值M
  每段权重w = exp(m - M)
  最终结果 = sum(w * n) / sum(w * d)
```

减去最大值是为了避免指数溢出。每段减去不同的最大值也没关系，因为第二阶段用`exp(m-M)`把它们
换回同一个尺度。这就是log-sum-exp合并。

## 内存与并行度

临时空间是`B × H × S × (D + 2) × 4 bytes`。DeepSeek B2/H12/S8/D128需要99,840 bytes，
远小于模型权重和KV cache。第一阶段blocks从24增加到192；第二阶段仍为24。

S最多32且不能超过sequence。每段至少包含一个token。接口支持FP32/BF16 cache、任意合法GQA
repeats和带更大capacity stride的dense prefix。

## 正确性门

公开实验接口是`cached_gqa_attention_split_sequence`。CPU走已有可读参考；HIP执行partial与combine
两个Kernel。它永远不会被普通`cached_gqa_attention`自动选择。

当前门覆盖：

- CPU小数值、splits 0和大于sequence的拒绝；
- PyTorch `softmax(QK^T * scale) @ V`完整输出；
- MI300X DeepSeek H12/KV2/D128；
- B1/B2、FP32/BF16、T31/32/33、T511/512/513/2048；
- 短序列S4、长序列S8；
- 完整context误差不超过8e-4；
- HIP计算区间0 payload H2D/D2H。

这些证据只说明“算法和接口目前算对了”。下一节点会搜索S1/2/4/8/16，并同时报告Event、wall、
逻辑/后端allocation和partial bytes。若T2048没有至少1.05倍收益，或者T512/B2反例过大，就拒绝
性能路线。
