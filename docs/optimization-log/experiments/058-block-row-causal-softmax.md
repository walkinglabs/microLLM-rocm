# Experiment 058 — cooperative causal-softmax rows

## 问题

Experiment 057 后，Attention 的最大两根柱子是 causal softmax forward `169.89 ms`
和 backward `108.89 ms`。旧 Kernel 让一个线程顺序扫描一整行。T=512 时，这个线程
需要独自做 max、exp、sum、归一化或梯度点积；同一个 block 里的其他线程在处理别的行。

## 假设与反驳条件

对长行使用一个 256-thread block：线程分摊列元素，用 shared-memory block reduction
求 max/sum/dot。短行保留旧实现，避免每行一个 block 的固定成本。

若 forward/gradient/因果屏蔽失败，任一模型收益低于 5%，peak 增加，T128 退化超过 5%，
或 profiler 没有同时降低前后向 Kernel 时间，则拒绝。

## 实现

只在 `T≥256` 路由：

```text
forward row:
  parallel max → parallel exp/sum → parallel normalize/zero future

backward row:
  parallel dot(dP, P) → parallel P × (dP - dot) / zero future
```

接口、dtype、矩阵布局、保存概率和 short-row Kernel 都没有改变。

## 正确性

定向门覆盖独立 softmax/RMSNorm 对照和完整 T=256 MHA/GQA Attention：输出、Q/K/V
梯度、未来 token 屏蔽及零 payload transfer 均通过。提交前还运行完整三层测试门。

## 正式模型结果

MI300X、BF16 Linear + FP32 master、batch 1、context 512、三进程中位数：

| 模型 | Experiment 057 | Experiment 058 | 自身加速 | measured peak | micro/PyTorch |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1634.49 | 2127.38 tok/s | 1.302× | 不变 | 0.258× |
| DeepSeek Distill 1.5B | 957.65 | 1145.36 tok/s | 1.196× | 不变 | 0.241× |

![Cooperative causal softmax](../assets/block-row-causal-softmax.svg)

T128 使用原 Kernel：`802.05→803.93 tok/s`（`1.002×`），peak 相同。

## Profiler

| 设备阶段 | 之前 | 之后 | 加速 |
|---|---:|---:|---:|
| causal softmax forward | 169.89 ms | 39.94 ms | 4.253× |
| causal softmax backward | 108.89 ms | 22.68 ms | 4.801× |
| 两者合计 | 278.78 ms | 62.62 ms | 4.452× |
| 全部 Kernel | 988.36 ms | 772.84 ms | 1.279× |

dispatch 精确保持 7631。HIP API 调用 `274847→259388`，但主解释无需依赖它：目标
Kernel 自身和全设备时间都大幅下降。

## 决定与新热点

保留。新 top-3 是 RMSNorm weight gradient `142.77 ms`、AdamW `126.82 ms`、bias
gradient `114.69 ms`。Attention softmax 已退出前三。下一节点应该优化 RMSNorm 权重梯度
的跨 row reduction，或用新 profile 划清 optimizer 同步记账边界。
