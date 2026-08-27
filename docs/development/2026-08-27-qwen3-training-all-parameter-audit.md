# Qwen3 全参数一步训练对齐

日期：2026-08-27

## 这一步为什么必要

上一节点只比较了 FFN 的 gate/up。它像检查了一栋楼里的两个房间：结果有价值，
但还不能说整栋楼都检查过。

这次把官方 Qwen3-0.6B 一次训练 step 中的全部独立参数都导出：

```text
forward 和 loss
→ backward
→ 在 AdamW 前保存全部梯度
→ AdamW 更新一次
→ 保存全部更新后参数
→ 与 PyTorch 同名、同 shape、逐值比较
```

## 名字和 shape 怎样对上

PyTorch Linear 权重通常保存为 `[输出, 输入]`，microLLM 内部使用 `[输入, 输出]`，
所以 Q/K/V/O、gate/up/down 和未绑权重的输出头要转置。Norm、Embedding 和一维 bias
不转置。

映射同时覆盖：

- Qwen3 的 Q/K-Norm；
- Qwen2/DeepSeek 使用的 Q/K/V bias；
- tied embedding；
- 未 tied 时单独存在的 `lm_head.weight`。

小 Qwen2 fixture 专门验证了 bias：14 个梯度和 14 个参数、各 680 个元素，名字、shape、
F32 payload 和连续 data offset 全部通过。gate/up 与 all 两种诊断导出互斥，避免重复写
大文件。

## tied 权重为什么是 311 变 310

checkpoint 里有 `model.embed_tokens.weight` 和 `lm_head.weight` 两个名字，但它们表示
同一份训练参数。运行时只更新一次，也只导出一次：

```text
311 个存储 Tensor
→ lm_head.weight 指向 token_embedding.weight
→ 310 个独立运行时 Tensor
→ 596,049,920 个独立参数值
```

梯度和一步后参数各比较一次，因此每种精度比较 1,192,099,840 个值。

## 固定门和结果

门在正式运行前写进审计器。Max 是所有元素中的最大绝对差；RMS 是全模型全部元素
合并后的 RMS。

| 精度 | Gradient Max/RMS | Parameter Max/RMS | 结果 |
|---|---:|---:|---|
| FP32 | `5.746e-4 / 5.024e-7` | `1.999e-5 / 5.110e-8` | 通过 |
| BF16 | `0.3641 / 4.071e-4` | `2.289e-5 / 2.253e-6` | 拒绝 |

FP32 的 loss 差是 `2.380e-7`。BF16 的 loss 差是 `0.009960`，Gradient Max 和
Parameter RMS 两道门失败。BF16 最坏梯度在 tied embedding；它不是只发生在 gate/up，
Attention QKV 和 FFN down 也有明显差异。

所有值在两种精度下都有限，名字、shape、Tensor 数和元素数也都正确。因此 BF16 的
结论是“完整运行但数值公式没有同步”，不是崩溃或漏参数。

## 一个不能隐藏的 RMS 口径

全模型 aggregate RMS 通过，不等于每个小 Tensor 的 RMS 都低于同一阈值。FP32 最坏
单 Tensor 梯度 RMS 是 `2.356e-5`，位于 block 5 Q-Norm；最坏参数 RMS 是
`1.206e-6`，位于 block 26 Attention Norm。raw 文件保留每个 Tensor 的数值，后续若要
增加 per-Tensor RMS 门，可以直接重新判定，不能把本节点误读成该门已经通过。

## 存储和性能边界

每种精度会临时产生四个 safetensors 文件，共约 9.54 GB。审计完成后文件已删除，仓库
只保存 620 条逐 Tensor 记录、聚合 summary 和 worker 元数据。导出时间不进入训练性能
结论；worker 明确标为 `diagnostic`。

本节点证明官方 FP32 的完整一步参数/梯度聚合门。AdamW moment、连续多步轨迹、SFT、
Radeon 和其他 ROCm 版本仍未由这份证据证明。

完整回归为 CPU 434/434、ASan/UBSan 431/431、MI300X HIP 215/215；coverage inventory
仍覆盖 199 个 Tensor 算子、45 个图 API 和 159 个注册测试文件。
