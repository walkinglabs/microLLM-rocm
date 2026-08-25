# 2026-08-25 — 先测 BF16 weight gradient，不先改训练图

当前 profile 显示训练主要时间仍在 GEMM。源码追踪进一步发现：BF16 Linear 只把 forward
变成低精度，weight gradient 仍是 FP32 `inputᵀ × output_gradient`。

新增的 benchmark-only 原型执行：

```text
FP32 input ──cast+transpose──> BF16 inputᵀ ┐
                                            ├─ BF16 GEMM ─> FP32 weight gradient
FP32 dY ───────────cast────────> BF16 dY ───┘
```

计时包含两次转换，不能把转换藏在 warm-up 外面。输出门分成两层：完整输出必须有限并与
FP32 基线报告 Max/RMS；64 个确定位置还要与 CPU 上的 BF16 数学核对。小形状
32×64×96 只有 0.823×，因此不存在全局默认开关。

下一步由 fresh-process matrix 测六个 Qwen/DeepSeek T512 真实 shape。只有两模型都出现
稳定的大 family 候选，才允许进入显式 Autograd 模型门。

