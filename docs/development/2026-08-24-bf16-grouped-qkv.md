# BF16 Grouped QKV：把三次投影放在一起

日期：2026-08-24

## 先用很简单的话解释

Attention 每层都要做 Q、K、V 三次矩阵乘法。它们读的是同一份输入，只是使用三份不同权重。
原来我们像让三辆车分别送货；GroupedGemm 像把三张订单交给同一个调度器。

但“少调用两次”不一定更快。我们测到：

- 如果每层都重新填写调度表，Qwen/DeepSeek 只有 `0.908×/0.815×`，反而更慢；
- 如果输入、输出和权重地址稳定，调度表只准备一次，算子达到 `1.881×/1.225×`。

所以关键能力不是一个开关，而是“指针稳定的 plan cache”。

## 为什么需要 QKV Arena

普通 Tensor 每次 forward 可能重新申请地址。Grouped plan 内部已经保存了指针，地址变化后不能继续用。
已有 QKV Arena 在所有 block 间复用同一块输入/输出空间，权重地址又长期不变。因此每个 block 第一次
建立自己的 plan，后续 forward 直接命中。key 还包含 shape、GPU、HIP/driver、hipBLASLt、device、
Stream 和所有指针，不能把旧 plan 错交给新 Tensor。

## 低精度输出的真实代价

当前 hipBLASLt 对这组 grouped shape 不支持直接 FP32 输出，只支持 BF16 输出。候选必须增加三次
BF16→FP32 cast。算子比较已经把 cast 算进时间；完整模型也检查所有 logits，而不是只看 top-1。

Qwen/DeepSeek logits Max/RMS 为 `0.09360/0.01978` 和 `0.06300/0.02044`，top token 不变，
满足事先写下的 BF16 门。峰值增加约 0.34%/0.17%。

## 为什么仍然不默认打开

Qwen T512 快 `3.17%`，DeepSeek 只快 `0.15%`。项目规则要求同一个不含模型名字的策略在两个模型
都至少快 `1%`。因此底层能力和显式 CLI 保留，默认关闭。以后如果更多 hidden≤1024 checkpoint
证明同一 shape 规律，才可以讨论 shape predicate；现在不能把“Qwen”写进 dispatch。
