# Experiment 079 — 缓存8个exp，softmax位级一致并加速

Experiment 077的last-logit profile里，causal softmax占Qwen设备时间26%。旧Kernel先把每线程
计算的`exp`写到全局output，规约出分母后再读回并归一化。

## 单变量候选

T≤2048时，每线程最多处理8个元素。候选把这8个`exp`暂存在局部数组，保持max、exp、本地
求和和block reduction顺序不变，分母就绪后只写一次最终概率；T>2048仍走旧Kernel。

反驳顺序按Experiment 078教训调整为：先完整logits，再测性能。

## 精度先通过

reference使用独立构建的`ef6fe1e`，候选使用同环境Release。Qwen/DeepSeek T2048 B8各比较
151,936个last logits：

```text
max abs = 0
RMSE    = 0
top     = equal
```

这证明寄存器值与原先写入再读取的FP32值位级一致，且规约顺序没有变化。focused softmax、
MHA/GQA和CPU/HIP测试也通过。

## 为什么第一份formal不能直接使用

第一轮候选三进程中，Qwen绝对吞吐从133k掉到115k/106k；但同一时段reference binary也从
历史约129k掉到92k–129k，而PyTorch保持稳定。跨窗口直接相除错误地得到0.889×。

因此保留该raw并标`invalid`，重新运行同GPU、同时间窗、交替顺序的双binary A/B。每一对只
改变softmax Kernel，避免把共享GPU漂移解释成候选效果。

## 配对正式结果：T2048 B8

| 模型 | reference median | register median | ratio of medians | median pair ratio |
|---|---:|---:|---:|---:|
| Qwen | 128,700 | 134,615 tok/s | 1.046× | 1.046× |
| DeepSeek | 66,189 | 67,604 tok/s | 1.021× | 1.022× |

Qwen有一对异常慢reference，但另外两对仍为1.042×/1.046×；DeepSeek三对稳定在
1.020×–1.023×。候选peak不变，它是纯吞吐优化。

## Profile与无spill证据

Qwen T2048 B8、一次warm-up加三次measured：

| 指标 | before | register | 变化 |
|---|---:|---:|---:|
| softmax | 130.719 ms | 111.299 ms | -14.86% |
| 全部Kernel | 501.814 ms | 481.707 ms | -4.01% |
| GEMM | 259.563 ms | 259.142 ms | 基本不变 |
| 无profile forward | 126.814 ms | 121.975 ms | 1.040× |

gfx942 code-object metadata显示：private segment=0，SGPR/VGPR spill均为0，LDS仍1024B；
资源只从46→48 SGPR、20→22 VGPR。softmax少19.42ms，全部Kernel少20.11ms，机制与端到端
方向一致。

## 边界shape

单对survey覆盖两模型、T256/512/1024/2048、B1/B8。除一条DeepSeek T512 B1外，15/16点
最差为0.987×，全部top token一致、peak不变。异常点0.830×与相邻shape矛盾，因此追加三对
复测，三个ratio为0.9935/0.9934/0.9941，未复现大回退且在5%门内。

T>2048仍保留旧Kernel，不扩大本轮支持边界。

HIP CTest还直接构造`[1,1,2048,2048]`分数，逐项与CPU比较并检查选定行的上三角严格为0、
可见概率和为1；它覆盖寄存器路径的最大边界，而不只依赖官方模型top token。

最终门：CPU 196/196、HIP 85/85、ASan/UBSan 194/194、Torch-enabled 199/199；优化日志与
测试覆盖validator通过。

![Register-cached causal softmax](../assets/register-softmax.svg)

## 决定

`keep`。位级精度、配对端到端、设备Kernel、寄存器资源和边界shape证据方向一致。第一份
cross-window失败与T512单点异常都保留并解释，没有删除不利数据。

下一节点读取新的last-logit profile；softmax仍是热点，但继续增加寄存器或改变规约顺序需要
新的完整logits门。更大的结构性差距仍是物化`[B,H,T,T]`，需要单独的online Attention实验。
