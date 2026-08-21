# Experiment 087 — 16个一组的缓存为什么会看“分配节奏”？

Experiment 086的token-history候选只少了21次小分配，却让DeepSeek B8 backend allocation从874
跳到13,863。这说明优化没有先撞上Attention，而是撞上allocator的相位。

## 旧池子的失败方式

旧exact-size pool把释放的块先放进`pending`：

```text
释放1个块 ┐
释放2个块 ├─ 还不能复用
...       │
释放16个块┘ → 记录一个Event → 才进入retired表
```

如果同尺寸Tensor在凑满16个之前又被申请，allocator看不见pending，只能`hipMalloc`。任何增加或
减少小Tensor的改动都会改变“第16个是谁”，所以性能依赖无关的分配数量。

## 为什么可以立即复用地址

这个pool有一个很强的既有合同：只允许legacy default Stream。GPU在同一Stream中按顺序执行：

```text
旧Kernel使用地址P
→ 新Kernel写地址P
```

即使CPU已经把P交给新Tensor，新Kernel也排在旧Kernel之后，不会提前覆盖。只要代码创建或传入
non-default Stream，`notify_non_default_stream()`就永久禁用pool，继续走真实alloc/free。

候选因此删除16-block Event批次，让释放块立即进入`retired[(device, exact_bytes)]`。它不做
size class、不跨尺寸复用、不提高8 GiB缓存上限，也不改变Tensor生命周期。

## 安全门

- 256次无中间同步的`fill → destroy → exact-size reuse`，最后值逐项正确；
- 16个同尺寸块无需completion batch即可16/16复用；
- non-default Stream仍永久禁用pool，alloc/free计数保持原合同；
- CPU 207/207、HIP 88/88、ASan/UBSan 200/200。

## DeepSeek T2048三对交替

| Batch | baseline tok/s | candidate tok/s | 速度比 | backend alloc | reuse |
|---:|---:|---:|---:|---:|---:|
| 1 | 66.60 | 67.23 | 1.010× | 1,091→94 | 15,232→16,229 |
| 8 | 496.01 | 512.39 | 1.033× | 903→94 | 15,423→16,232 |

两边peak完全不变，candidate backend deallocation为0，三对token全部一致。更重要的是candidate
三轮allocation固定为94；基线B8仍有一轮再次爆到13,884并跌至436 tok/s。

## T512 B8反驳实验

宽矩阵的单进程曾显示Qwen T512 B8回退9.8%，所以不能直接接受。三对交替复测：

| 模型 | baseline median | candidate median | 速度比 | backend alloc |
|---|---:|---:|---:|---:|
| Qwen | 1,676.16 | 1,700.00 | 1.014× | 797→86 |
| DeepSeek | 905.21 | 994.72 | 1.099× | 958→94 |

Qwen第一轮baseline只有1,002 tok/s，后两轮约1,677；candidate三轮稳定在1,692–1,715。单轮负面
结论被推翻，正好说明为什么小差异必须交替多进程。

## 官方shape survey

Qwen/DeepSeek的T8/T512/T2048、B1/B8共24条candidate/PyTorch进程全部成功。Qwen 6/6 token
一致；DeepSeek T8/T512一致，T2048两点保留已有分叉。candidate所有shape只有82–94次backend
allocation、0次backend deallocation，peak与Experiment 085逐shape相同。

![Immediate default-stream exact-size pool](../assets/immediate-default-stream-pool.svg)

## 决定

保留candidate。它解决的是allocator确定性，不是Attention数学：

1. 分配数量变化不再改变16-block retirement相位；
2. 默认Stream异步顺序由压力测试证明；
3. non-default Stream安全边界没有放宽；
4. T2048和T512关键shape没有稳定回退；
5. active peak不变，reserved/cached仍单独报告。

Experiment 088回到设备第一热点：只优化`cached_attention_fused_kernel`的BF16 K/V读取与规约。

数据见[`087-data`](087-data/)。
