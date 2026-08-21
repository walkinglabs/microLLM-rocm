# Experiment 080 — 不物化T²，不等于FlashAttention

Experiment 079后，结构性问题仍是`QK → [B,H,T,T] → softmax → PV`。开写新Kernel前先盘点
本机ROCm：安装了rocWMMA header/CMake package，但没有Composable Kernel或可直接链接的
FMHA runtime。因此没有一个“加一行库调用”就能使用的FlashAttention实现。

## 最小反驳实验

仓库已有一个不物化全局T²的可读fused Attention Kernel。它按query row使用一个block，分数
留在shared memory；如果“只要不物化T²就更快”，把长inference从hipBLASLt+softmax路由到
这个Kernel至少不应大幅回退。

先只测Qwen T512 B1，两个binary在同GPU交替两对：

| 路径 | 两次 tok/s | median | peak |
|---|---:|---:|---:|
| library QK/softmax/PV | 93,570 / 92,947 | 93,259 | 1.216 GiB |
| readable fused | 33,582 / 33,581 | 33,581 | 1.196 GiB |

两对比值为0.359×/0.361×，回退约64%，peak只下降约1.7%；top token仍一致。失败已经远超
门，因此不继续浪费T2048 B8资源。

![Readable fused Attention discarded](../assets/readable-fused-attention-discard.svg)

## 为什么失败

这个Kernel避免全局score Tensor，却由普通线程顺序做head-dimension点积和PV累计，没有使用
MFMA/rocWMMA矩阵片段，也没有tile级online softmax的数据复用。当前hipBLASLt路径虽付出T²
存储，但矩阵乘法效率高得多。

Experiment 056早已在T512 profile过同一因果关系；本轮在修正后的last-logit/BF16/B1语义下
重新验证，结果仍一致。

## 决定

`discard`，路由改动回退，原始paired数据保留在[`080-data`](080-data/)。下一版真正的online
Attention需要rocWMMA/MFMA tile、online max/sum、因果边界和GQA共享设计，属于独立大型节点，
不能把现有可读Kernel改名为FlashAttention。
