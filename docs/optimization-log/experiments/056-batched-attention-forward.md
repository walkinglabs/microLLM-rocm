# Experiment 056 — batched long-sequence Attention forward

## 问题

Experiment 055 已经不在 backward 重算 softmax，但 forward 仍由一个容易阅读的
HIP Kernel 完成。这个 Kernel 中，一个线程顺序扫过很多 key。序列越长，它重复做的
点积越多。Qwen `T=512` 的 72 次 forward 一共需要 `272.52 ms`。

## 假设与反驳条件

假设：把 `QKᵀ` 和 `PV` 当成很多组小矩阵，交给 Experiment 053 已验证的
strided-batched hipBLASLt，会让两个真实模型变快。

出现任意一种情况就拒绝：

- MHA 或 GQA 的输出、Q/K/V 梯度不再与 CPU reference 对齐；
- Qwen 或 DeepSeek `1×512` 没有至少 5% 的三进程中位数收益；
- measured peak 增加；
- `T=128` 退化超过 5%；
- profiler 只显示 host 计时变化，设备 Kernel 时间没有下降。

## 只改变什么

仅在 HIP、`T≥256`、head width 不超过已有上限且 hipBLASLt 可用时：

```text
Q --scale--┐
           ├─ batched QKᵀ ─ causal softmax ─ batched PV ─ output
K --repeat─┘                         │
V --repeat───────────────────────────┘
                                    └─ saved probability for backward
```

短序列仍用原来的 fused Kernel；没有 hipBLASLt 的构建仍能使用 composed/readable
实现。没有更改 dtype、loss、optimizer、权重或 PyTorch 对照协议。

## 正确性

现有 `HipFullAttentionTest` 覆盖 MHA/GQA、`T=1/3/32/128/256`、forward、保存概率后的
backward、因果上三角以及零 payload transfer。候选先通过 `T=256` 定向测试；完整门在
提交前再次运行。

## 正式结果

MI300X、BF16 Linear + FP32 master、batch 1、context 512、每框架三个新进程：

| 模型 | Experiment 055 | Experiment 056 | 自身加速 | measured peak | micro/PyTorch |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1248.17 | 1361.17 tok/s | 1.091× | 不变 | 0.170× |
| DeepSeek Distill 1.5B | 627.83 | 731.34 tok/s | 1.165× | 不变 | 0.156× |

![Batched Attention forward](../assets/batched-attention-forward.svg)

`T=128` 不进入新分支：`802.47→812.36 tok/s`，peak 完全相同，满足回退门。

## Profiler 解释

Qwen retained process：

| 项目 | 之前 | 之后 |
|---|---:|---:|
| 旧 fused forward / 新完整 forward stage | 272.52 ms | 178.29 ms |
| 其中 causal softmax | — | 169.93 ms |
| 其中两次 batched GEMM | — | 1.75 ms |
| 其中 K/V repeat | — | 5.41 ms |
| 全部 Kernel 时间 | 1283.85 ms | 1185.53 ms |
| Kernel dispatch | 7055 | 7343 |
| HIP API calls | 259593 | 266578 |

forward stage 是 `1.528×`，全进程 Kernel 是 `1.083×`。launch 和 API 数量反而增加，
说明收益来自把重复标量点积交给矩阵硬件，不是把等待藏到别处。

## 决定

保留。两个模型、短序列回退、显存和设备时间证据方向一致。

新的最大热点是 saved-row backward `306.63 ms`，其次是 causal softmax `169.93 ms`、
RMSNorm weight gradient `143.08 ms`、AdamW `128.86 ms` 和 bias gradient `117.84 ms`。
下一节点应从这些真实柱子选择一个变量，不能继续用“Attention 一定最慢”代替测量。
