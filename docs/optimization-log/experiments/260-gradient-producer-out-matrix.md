# Experiment 260 — producer直接写最终地址，能否同时删掉临时Tensor和leaf add

Status: `admitted to scoped Autograd gate`

Baseline执行allocating `input^T @ dY`再做leaf add；candidate直接写caller-owned output。
四个Model-S真实shape和一个tiny反例，每shape三个fresh process、5次warm-up、40次测量，轮换
operation与shape顺序。

| Shape | Event | Wall | Allocation |
|---|---:|---:|---:|
| head T32 | 1.873× | 1.612× | 1→0 |
| FFN T32 | 1.260× | 1.181× | 1→0 |
| Attention T32 | 1.179× | 1.122× | 1→0 |
| head T512 | 1.426× | 1.363× | 1→0 |
| tiny反例 | 1.178× | 1.101× | 1→0 |

![Gradient producer out matrix](../assets/gradient-producer-out-matrix.svg)

15个完整输出位级一致，CPU/HIP/PyTorch对齐；全部shape同时过Event/Wall 1.05门。这个结果只
准入scoped Autograd：必须证明仅在“预设为零、尚无贡献、right leaf、rank-2”时覆盖写，否则
恢复普通accumulate。没有创建模型或DDP route。

证据：[`producer matrix`](../../../benchmarks/results/2026-08-25-gradient-producer-out-matrix/)
