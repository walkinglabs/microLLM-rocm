# Experiment 276 — B1和B2的梯度为什么不能直接平均

Status: `explicit token-weighted mode kept`

tiny两rank固定rows `[1,2]`、context4。rank0有4 token，rank1有8 token。

| Gate | Result |
|---|---:|
| equal-only | 2/2进程明确失败 |
| average tokens | 6 |
| rank0/rank1 scale | 0.666666687 / 1.333333373 |
| 3-step rank Max/RMS | 0 / 0 |
| CPU parameter Max/RMS | 8.18e-8 / 8.79e-9 |
| weighted loss diff | 1.94e-7 |

![Ranked input weighting](../assets/ranked-input-weighting.svg)

每rank loss/gradient都是local mean。先乘`local_tokens/average_tokens`再做RCCL average，代数上等于
`sum(local_grad*local_tokens)/global_tokens`。CPU B3拼接batch完整参数门验证这个公式。

默认equal-only在参数collective前拒绝不等token数，防止静默错误。token-weighted显式保留。
ready overlap暂不支持weighted，因为hook可能在post-backward scaling前已enqueue bucket。

下一节点扩展Model-S `[1,2]` one-step smoke，确认大参数/三bucket同步路径；不同时解决weighted
overlap。

证据：[`ranked input weighting`](../../../benchmarks/results/2026-08-25-ranked-input-weighting/)
