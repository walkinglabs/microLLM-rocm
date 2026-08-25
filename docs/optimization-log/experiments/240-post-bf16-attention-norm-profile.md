# Experiment 240 — 两个Norm融合后，剩余cast恰好一进一出

Status: `profile complete; attribution required`

## 新默认地图

四个进程同时要求FFN Norm和Attention Norm默认启用，仍用`(six-one)/5`。

| Model | Kernel ms prior | Kernel ms both | Cast calls prior→both | GEMM share |
|---|---:|---:|---:|---:|
| Qwen | 8.208 | 8.069 | 72→48 | 61.5% |
| DeepSeek | 14.659 | 14.489 | 84→56 | 68.8% |

cast又精确减24/28。剩余每层恰好两次：一次FP32→BF16，一次BF16→FP32。

![Post Attention Norm profile](../assets/post-bf16-attention-norm-profile.svg)

## 为什么先不写代码

FP32→BF16很可能是Attention context进入BF16 O projection；BF16→FP32很可能是grouped V
回到当前FP32 Attention core。但只有kernel名称还不足以证明具体source。下一节应先使用
已有allocation/operation边界或定向计数器将两次cast归因，然后只选其中一个。
