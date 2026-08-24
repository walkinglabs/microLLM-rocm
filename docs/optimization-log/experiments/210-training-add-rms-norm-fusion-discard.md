# Experiment 210 — 训练残差加法与RMSNorm融合

Status: `discard model route; keep primitive`

## 先把问题说简单

Transformer的一层里，Attention算完后要做两件事：

1. 把Attention结果加回原输入，得到“残差和”；
2. 把残差和归一化，再交给FFN。

旧路径启动一个加法Kernel，再启动一个RMSNorm Kernel。仓库已有一次完成两件事的
`ops::add_rms_norm`，所以候选尝试在训练图里复用它。

## 为什么不能只返回归一化结果

残差和有两条用途：一条直接走到本层输出，另一条经过RMSNorm和FFN。反向传播时，这两条路的
梯度必须先在残差和节点相加，再把同一份总梯度交给加法两边。候选因此保留两个图节点：

```text
left ─┐
      ├─ add_rms_norm_sum ─┬─ residual path
right ┘                    └─ add_rms_norm ─ normalization path
                                      └─ weight gradient
```

这不会减少数学节点，只减少GPU启动。CPU组合参考、HIP设备图和PyTorch分别检查两个前向输出，
以及left、right、weight的全部梯度。HIP执行期间H2D/D2H都是0。

## 正式门

同一临时二进制只切换一个开关。两个模型都是BF16 Linear + FP32 master，B1/T512，一次热身、
两步训练、每个策略三个新进程，顺序交替。

| Model | Materialized | Fused | Speedup | Peak | Loss relative diff | Parameter equal |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 15,234.08 | 14,906.34 tok/s | 0.9785× | 1.000× | 0.0679% | yes |
| DeepSeek Distill 1.5B | 6,324.03 | 6,311.17 tok/s | 0.9980× | 1.000× | 0.0073% | no |

两个模型都没达到`1.01×`。DeepSeek的固定参数还从`2.124970913`变成`2.124971151`。这不是
发散，但它证明融合改变了浮点运算路径，不能写成“完全相同”。

## Profile反驳了什么

Qwen的结构profile证明路由确实生效：

| Counter | Materialized | Fused | Delta |
|---|---:|---:|---:|
| All Kernel calls | 6,903 | 6,831 | -72 |
| FP32 add calls | 504 | 432 | -72 |
| standalone RMSNorm forward | 147 | 75 | -72 |
| fused add+RMSNorm | 0 | 72 | +72 |
| Total Kernel time | 109.741 ms | 109.692 ms | -0.045% |

少72次launch是真的，但只节省约0.049 ms；其他Kernel的正常抖动已经能盖住它。这个实验推翻
“只要少一次逐元素Kernel，整步训练就会稳定更快”的解释。

![Training add plus RMSNorm discard](../assets/training-add-rms-norm-discard.svg)

## 决定

删除Transformer和CLI的候选路由，不改变训练默认路径。保留`autograd::add_rms_norm`原语及其
CPU/HIP/PyTorch测试，供未来更大范围的图融合器使用。下一次训练优化必须先从新profile选择
占比更高的工作，或者一次融合完整残差分支，不能继续追逐单个微小launch。

原始证据在
[`benchmarks/results/2026-08-24-training-add-rms-norm-fusion/`](../../../benchmarks/results/2026-08-24-training-add-rms-norm-fusion/)。
