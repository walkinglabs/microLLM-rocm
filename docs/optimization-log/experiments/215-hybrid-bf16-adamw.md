# Experiment 215 — 小书合订，大书单放

Status: `keep`

## 为什么 Experiment 214 只到 partial keep

BF16 moment 已把两份状态减半，大 Tensor Kernel 也明显更快。但 Qwen 有许多很小的参数 Tensor，
逐个 launch 的固定成本让 optimizer 只有 `1.0687×`，没过 `1.10×` stretch gate。把所有 Tensor
全部合并又在旧实验中伤害 DeepSeek。

新假设是：

```text
小 Tensor ─┐
小 Tensor ─┼─ 一个 multi-tensor Kernel
小 Tensor ─┘

大矩阵 ───── 各自使用 BF16-moment 向量 Kernel
embedding ── 各自使用 BF16-moment 向量 Kernel
```

这像把许多薄练习册合订，大课本仍单独放。阈值只改变提交方式，不改变 moment dtype、公式或
checkpoint，因此数值结果与逐 Tensor BF16 路线一致。

## 阈值不是拍脑袋

我们扫描 4K、64K、256K、1M、4M、16M。4K–64K 在两个模型上选择集合相同，不能把时序噪声
解释成阈值收益。1M 首次覆盖足够多的中等 Tensor，又避开最大的矩阵和 embedding。

| Threshold | Qwen optimizer | DeepSeek optimizer | Qwen E2E | DeepSeek E2E |
|---:|---:|---:|---:|---:|
| 4K | 1.155× | 1.239× | 1.033× | 1.031× |
| 64K | 1.151× | 1.243× | 1.042× | 1.039× |
| 256K | 1.188× | 1.226× | 1.043× | 1.035× |
| 1M pilot | 1.235× | 1.268× | 1.062× | 1.055× |
| 4M | 1.222× | 1.194× | 1.069× | 1.036× |
| 16M | 1.063× | **0.896×** | 1.033× | **0.980×** |

16M 把 DeepSeek 337 个 Tensor、1.31B 元素交给通用 Kernel，形成明确反例。两模型端到端几何
收益在 1M 比 4M 更高，因此 1M 进入正式门。

## 五进程正式结果

| Model | FP32 moment tok/s | Hybrid BF16 tok/s | E2E | Optimizer | Peak ratio |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,877.53 | 15,605.96 | 1.0490× | 1.2404× | 0.8329× |
| DeepSeek Distill 1.5B | 6,204.38 | 6,532.08 | 1.0528× | 1.2631× | 0.8084× |

两模型的 loss、moment bytes、峰值、端到端和 optimizer stretch 门全部通过。相对 Experiment
214 的逐 Tensor BF16 route，又提高 `1.0245×/1.0185×`。

![Hybrid BF16 AdamW](../assets/hybrid-bf16-adamw.svg)

## 保留边界

- 整个框架默认仍是 FP32 moments；
- 一旦用户显式选择 BF16 moments，HIP Auto 使用 1M；
- `--adamw-bf16-multi-tensor-threshold 0` 可关闭合并；
- 正数只用于可复现实验，不是跨 GPU 的普遍最优值；
- checkpoint 不记录阈值，因为完整状态和后续数值轨迹不随 dispatch 改变。

原始数据在
[`benchmarks/results/2026-08-24-hybrid-bf16-adamw/`](../../../benchmarks/results/2026-08-24-hybrid-bf16-adamw/)。
