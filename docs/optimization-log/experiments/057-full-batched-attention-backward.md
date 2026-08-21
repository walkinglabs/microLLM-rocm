# Experiment 057 — fully batched saved Attention backward

## 问题

Experiment 056 后，最大的单个 Kernel 是 saved-row backward：72 次共 `306.63 ms`。
它已经使用 forward 保存的概率，却仍在每个 query row 中用普通循环计算：

```text
dP = dO · Vᵀ
dS = softmax_backward(P, dP) × scale
dQ = dS · K
```

`dK` 和 `dV` 已经由 batched GEMM 完成，因此同一 backward 里一半像矩阵库，另一半
仍像手写三重循环。

## 假设与拒绝线

把 `dP` 和 `dQ` 也交给 Experiment 053 的 strided-batched hipBLASLt，只留下 causal
softmax backward。若任一 Q/K/V 梯度不对、任一真实模型收益低于 5%、measured peak
增加、T128 回退超过 5%，或设备时间没有下降，就拒绝。

## 实现

仅在 HIP、`T≥256` 且 hipBLASLt 可用时：

```text
dP = batched(dO, Vᵀ)
dS = causal_softmax_backward(P, dP) × scale
dQ = batched(dS, K)
dK = reduce_gqa(batched(dSᵀ, Q))
dV = reduce_gqa(batched(Pᵀ, dO))
```

GQA 先使用已有 `repeat_interleave` 展开 K/V heads，最后用已有 deterministic reduction
合并 dK/dV。短序列、CPU 和 library-free 路径不变。

## 正确性

`HipFullAttentionTest` 覆盖 MHA/GQA、T=256、完整输出、Q/K/V 梯度、因果屏蔽和零 host
payload transfer。候选先过该定向门，再运行正式模型；提交前还需通过完整、Sanitizer 和
PyTorch-enabled 三层门。

## 正式结果

MI300X、BF16 Linear + FP32 master、batch 1、context 512、三进程中位数：

| 模型 | Experiment 056 | Experiment 057 | 自身加速 | measured peak | micro/PyTorch |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1361.17 | 1634.49 tok/s | 1.201× | 不变 | 0.190× |
| DeepSeek Distill 1.5B | 731.34 | 957.65 tok/s | 1.309× | 不变 | 0.198× |

![Fully batched Attention backward](../assets/full-batched-attention-backward.svg)

T128 在未改路径上 `812.36→802.05 tok/s`，即 `0.987×`，peak 完全相同。单次波动没有
越过 5% 拒绝线。

## Profiler 解释

旧 `306.63 ms` saved-row Kernel 消失。替代它的是：

- causal softmax backward：`108.89 ms`；
- 新增 dP/dQ batched GEMM：约 `1.67 ms`；
- backward K/V head repeat：约 `5.45 ms`；
- scaled dS：约 `6.21 ms`。

替代阶段合计约 `122.21 ms`，即 `2.509×`；全进程 Kernel 时间
`1185.53→988.36 ms`，即 `1.199×`。dispatch 与 HIP API 仍分别增加约 3.9%/3.1%，
再次排除“少 launch”解释。

## 决定与下一问题

保留。两个模型、设备 trace、显存、数值和 fallback 证据一致。

新 top-5 是 causal softmax forward `169.89 ms`、RMSNorm weight gradient
`142.93 ms`、AdamW `126.65 ms`、bias gradient `114.78 ms` 和 causal softmax backward
`108.89 ms`。下一实验应优化 softmax row 并同时覆盖 forward/backward，而不是继续保存
更多 T² 表。
