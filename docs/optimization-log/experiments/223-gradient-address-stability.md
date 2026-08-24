# Experiment 223 — 同一个参数，下一次backward还是同一个gradient地址吗

Status: `keep diagnostic; model/context-specific Graph eligibility`

## 为什么必须先测地址

两节点AdamW Graph的descriptor保存真实gradient地址。`zero_grad()`会让参数忘记旧gradient，
下一次backward重新产生Tensor。即使shape和值一样，新Tensor也不保证拿到同一块Storage。

本实验不改allocator。每个进程执行：

```text
一次warmup backward
→ 打开当前训练CLI使用的exact-size default-Stream pool
→ measured backward A，记住每个参数gradient Storage身份
→ zero_grad
→ measured backward B，只报告相同/变化，不导出原始指针值
```

Qwen/DeepSeek使用仓库固定的正式架构配置和finite synthetic weights。地址生命周期由shape、图和
allocator决定，不把synthetic结果写成真实loss/模型效果证据。

## 18进程正式矩阵

| Model | Precision | Context | Stable tensors | Changed tensors | Changed bytes |
|---|---|---:|---:|---:|---:|
| Tiny GQA | FP32 | 8 | 17/21 | 4 | 8,192 |
| Tiny GQA | BF16 | 8 | 17/21 | 4 | 8,192 |
| Qwen 0.5B config | BF16 | 8 | 290/290 | 0 | 0 |
| Qwen 0.5B config | BF16 | 512 | 290/290 | 0 | 0 |
| DeepSeek-Distill 1.5B config | BF16 | 8 | 339/339 | 0 | 0 |
| DeepSeek-Distill 1.5B config | BF16 | 512 | 141/339 | 198 | 7,107,772,416 |

![Gradient Storage address stability](../assets/gradient-address-stability.svg)

三个新进程中，每个case变化的参数名集合完全一致，不是一次地址随机抖动。

## 变化发生在哪里

Tiny的四项固定是两层的K/V projection weight gradient。

DeepSeek T512变化集合为：

- Attention 112项：28层×Q/K/V/O；
- FFN 84项：28层×gate/up/down；
- embedding/head 2项；
- 稳定的141项主要是norm与bias，只占579,584字节；
- 变化地址覆盖7,107,772,416字节，几乎是全部大gradient payload。

DeepSeek T8却339/339稳定；因此不能只按`model=deepseek`决定Graph eligibility。context改变了中间
Tensor大小、回收顺序和exact-size池的LIFO复用结果。

## 决定

- 保留不导出原始指针的gradient address benchmark、三进程runner与schema测试；
- Qwen T8/T512和DeepSeek T8允许进入下一轮optimizer-phase Graph性能门；
- DeepSeek T512禁止复用当前immutable descriptor；只能每步recapture，或先建立stable gradient
  buffer；
- Graph workspace未来必须绑定一次实际gradient snapshot，而不是只绑定model/config；
- 下一节点比较Qwen T512与DeepSeek T8的eager/multi-Graph optimizer phase，同时把DeepSeek
  T512作为显式拒绝case，不做未定义行为。

原始证据位于
[`benchmarks/results/2026-08-24-gradient-address-stability/`](../../../benchmarks/results/2026-08-24-gradient-address-stability/)。
