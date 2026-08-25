# Experiment 243 — 当前推理局部搜索到边界了吗

Status: `complete, local policy search saturated`

把 GPU 想成一条流水线。现在每层还剩两次“换包装”：一次把 FP32 变成 BF16，
一次再把 BF16 变回 FP32。真实 profile 显示，它们只占 Qwen 2.694%、DeepSeek
1.841% 的 Kernel 时间。

即使出现一个不花任何时间的魔法，把剩余转换全部删掉，Kernel-only 加速上限也只有：

| Model | Remaining cast share | Perfect-deletion ceiling |
|---|---:|---:|
| Qwen2.5-0.5B | 2.694% | 1.0277× |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.841% | 1.0188× |

![Current inference local saturation](../assets/inference-local-saturation.svg)

最近六条范围明确的路线分别被完整模型反例或后端能力门关闭：online Attention、
exact Attention solution、vectorized SwiGLU、grouped Swish，以及 P×V 两个方向的
mixed-dtype 接口。这说明继续拨动同一批局部开关，预期收益已经小于测量噪声和维护成本。

这不是“推理再也不能优化”。它只关闭当前默认路径上的局部策略搜索。下一份实现合同必须
改变架构尺度，例如新 custom kernel、跨算子图融合或新的后端/硬件矩阵；不能换一个线程数
就重复宣称新路线。

证据：[`saturation package`](../../../benchmarks/results/2026-08-25-inference-local-saturation/)

