# Experiment 136：可归因前向再降20%，热点转向GEMM

| 模型 | dynamic calls | scale三段 | 降幅 | GEMM | known-forward | 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 168→96 | 2.122→1.154ms | **45.6%** | 3.013ms | 5.241→4.167ms | **20.5%** |
| Deep | 197→113 | 3.109→1.755ms | **43.6%** | 5.401ms | 8.628→7.155ms | **17.1%** |

![Shared activation profile](../assets/fp8-shared-activation-profile.svg)

Qwen/Deep kernel与API launches分别少216/252，等于减少的72/84 dynamic调用×3个Kernel。GEMM
calls、other calls、fallback均不变，收益可以隔离归因。

共享后GEMM占known-forward约72%/75%。但Exp135中Deep已经1.028× BF16，Qwen为0.923×；
同时完整RMS仍为门的5.85×/4.98×。因此此时继续做复杂GEMM autotune不是最重要的系统缺口，
下一节点回到精度：分析逐层误差贡献或校准，而不是用更多性能代码掩盖模型不可用。
