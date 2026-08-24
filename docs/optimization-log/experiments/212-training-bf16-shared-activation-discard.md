# Experiment 212 — 同一份激活不要重复转成BF16

Status: `discard model routes; keep primitives`

## 先分清“加载”和“训练”

当前profile里，DeepSeek的`cast_transpose_2d`占7.9%，看起来很大。但它的197次调用恰好等于
加载时需要转置的权重数量，全部发生在checkpoint加载阶段，不能把它当作每步训练热点。

真正重复的是`cast_kernel<float, bfloat16>`。每个BF16 Linear都把自己的FP32输入转一次BF16。
Q、K、V读的是同一个Attention输入，却转三次；gate、up读的是同一个FFN输入，却转两次。

## 多输出图怎样保持梯度

共享cast只发生在前向。每个输出仍是独立图节点：

```text
FP32 input ─ cast once ─┬─ Q / gate GEMM ─ own weight edge
                       ├─ K / up GEMM   ─ own weight edge
                       └─ V GEMM        ─ own weight edge
```

反向仍使用FP32 master input和weight。每个输出分别算input gradient和weight gradient，input节点
负责累加所有分支。CPU组合图、PyTorch公式和HIP设备图都比较了每个输出、input gradient以及
五组weight gradient；HIP测量区间H2D/D2H均为0。

## 结构profile

两模型都是加载、一次热身和两步测量：

| Model | Separate casts | Shared casts | Removed | Cast time | All Kernel speedup |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 672 | 456 | 216 | 4.052→2.783 ms | 1.0116× |
| DeepSeek Distill 1.5B | 788 | 536 | 252 | 9.266→6.473 ms | 1.0095× |

216/252正好等于三步里每层每步少72/84次cast。目标工作确实消失，但总Kernel收益只有约1%。

## 三条模型反驳

| Policy | Qwen | DeepSeek | Processes | Decision |
|---|---:|---:|---:|---|
| QKV + gate/up | 1.0066× | 1.0179× | 20 | reject |
| QKV only | 0.9804× | 1.0039× | 12 | reject |
| gate/up only | 0.9911× | 1.0012× | 12 | reject |

组合路径先跑三进程，Qwen只有1.0062×；因为profile证明结构命中，我们没有降门，而是扩大到
五进程。Qwen仍只有1.0066×。再拆开两个边界后，两条独立路线也都失败，说明组合里的DeepSeek
收益不能推广成稳定的共享cast策略。

![Training BF16 shared activation discard](../assets/training-bf16-shared-activation-discard.svg)

## 决定

删除Transformer、optimizer和CLI路由，不改变默认训练。保留`bf16_gate_up_projection`、
`bf16_qkv_projection`的多输出Autograd接口和CPU/HIP/PyTorch测试。下一次若重用它们，应由能
同时组合GEMM提交或规划整个图的系统调用，不能只为少一个cast改变eager执行顺序。

原始证据在
[`benchmarks/results/2026-08-24-training-bf16-shared-activation/`](../../../benchmarks/results/2026-08-24-training-bf16-shared-activation/)。
