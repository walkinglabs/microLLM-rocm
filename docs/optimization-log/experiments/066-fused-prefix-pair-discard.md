# Experiment 066 — 删除所有复制，为什么长 batch 反而更慢

## 假设

Experiment 065的BF16 full prefill先把Key和Value各cast一次，再按`batch×head`复制进带
capacity stride的Cache。Qwen T2048 B8每个正式进程的measured区有3840次D2D copy。

候选让一个HIP Kernel同时完成：

```text
读取FP32 Key和Value
→ 转成BF16
→ 计算batch/head/token/column目标位置
→ 直接写入两块capacity-strided Storage
```

直觉很诱人：少96次cast、全部D2D归零，应该更快。固定反驳条件是：Qwen/DeepSeek、
T32/512/2048、B1/B8中任一关键shape的prepare或端到端稳定退化，就不能无条件替换。

## 正确性

CPU/HIP smoke覆盖FP32/BF16、B2、capacity大于active sequence、错误stride和零host payload
transfer。官方12-shape完整logits与Experiment 065相同：11/12通过，原有DeepSeek T512 B1
RMSE失败不变；12/12 suffix、finite、top token和Cache字节都不变。

候选没有制造新数值错误，但正确不等于应当保留。

## Release正式矩阵

72/72进程成功。下表的prepare比值大于1才是候选更快：

| 模型 | T | B | prepare ms 065→候选 | prepare比 | E2E比 | decode比 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 32 | 1 | 7.715→7.504 | 1.028× | 0.996× | 0.992× |
| Qwen | 32 | 8 | 6.478→4.400 | 1.472× | 1.056× | 1.019× |
| Qwen | 512 | 1 | 5.993→5.747 | 1.043× | 1.004× | 1.001× |
| Qwen | 512 | 8 | 22.665→20.742 | 1.093× | 1.027× | 1.008× |
| Qwen | 2048 | 1 | 20.412→20.577 | 0.992× | 0.992× | 0.993× |
| Qwen | 2048 | 8 | 400.372→522.553 | **0.766×** | **0.826×** | 1.054× |
| DeepSeek | 32 | 1 | 5.974→5.735 | 1.042× | 1.002× | 0.999× |
| DeepSeek | 32 | 8 | 10.224→8.222 | 1.243× | 1.026× | 1.006× |
| DeepSeek | 512 | 1 | 10.525→10.262 | 1.026× | 1.002× | 1.000× |
| DeepSeek | 512 | 8 | 48.470→46.712 | 1.038× | 0.984× | 0.964× |
| DeepSeek | 2048 | 1 | 34.899→34.428 | 1.014× | 0.999× | 0.997× |
| DeepSeek | 2048 | 8 | 250.880→250.547 | 1.001× | 0.994× | 0.989× |

![Fused prefix pair discarded](../assets/fused-prefix-pair-discard.svg)

候选36条microLLM原始进程的measured D2D全部是`0 calls / 0 bytes`。直接目标达成了，
但Qwen T2048 B8 prepare慢30.5%，端到端慢21.1%。候选三轮prepare分别为
`522.553/540.997/522.254ms`，不是一个离群点。

decode没有被本节点修改，其正负变化是共享GPU时段波动，不能拿Qwen长shape的decode上涨
抵消prepare失败。

## 为什么短profile会误导

同一个Qwen T2048 B8的一次warm-up、一次measured profile显示：

| 指标 | Experiment 065 | 候选 |
|---|---:|---:|
| Kernel总时间 | 316.195ms | 309.213ms |
| Kernel calls | 6575 | 4991 |
| FP32→BF16 cast | 840 / 9.546ms | 744 / 8.556ms |
| prefix-pair | 无 | 48 / 0.639ms |
| measured D2D | 768 / 201.33MB | 0 / 0 |
| 单次Cache prepare | 136.944ms | 133.297ms |

这张局部证据完全支持候选。但正式协议重复五次step并取三个新进程中位数，稳定暴露了相反
结果。新增Kernel本身只有0.639ms，无法解释多出的约122ms；更可能的下一解释是临时Tensor
生命周期、allocator reuse/retirement或重复prefill状态，而不是“Kernel算得太慢”。

## 决定

**discard**。删除prefix-pair公共API、Kernel、模型路由和测试，保留Experiment 065的
cast+per-head copy路径。只保留一个独立合同修复：paired step store现在明确拒绝Key/Value
输入dtype不一致，避免HIP把BF16地址当成FP32读取。

下一次若重试，必须先有独立prefix microbenchmark和逐step allocator计数，并用同一binary
显式切换两条路径；不能再凭“copy归零”和一次profile直接合入。
