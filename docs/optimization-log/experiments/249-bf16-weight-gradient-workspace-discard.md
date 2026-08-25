# Experiment 249 — 预分配为什么没有更快

Status: `workspace API rejected`

exact-size cache预热后，同一进程分别测公共allocating API与数学等价的preallocated组成路径。
Event只看设备工作，wall还包含host Tensor构造和cache lookup。

| Model | Event preallocated/allocating | Wall | Minimum wall |
|---|---:|---:|---:|
| Qwen | 1.037× | 0.986× | 0.956× |
| DeepSeek | 0.886× | 0.889× | 0.840× |

![BF16 weight-gradient workspace discard](../assets/bf16-weight-gradient-workspace-discard.svg)

公共API每次精确3次cache reuse、0次backend allocation。Qwen Event略好但wall未过门；
DeepSeek的preallocated Event与wall都稳定更慢。0/2 shape通过wall median 1.01且minimum 1.0门。

因此不创建 `Bf16WeightGradientWorkspace` 或 out API。allocation calls可见不代表它们是性能瓶颈。
独立算子和成本runner保留，模型route继续不存在。

证据：[`workspace gate`](../../../benchmarks/results/2026-08-25-bf16-weight-gradient-workspace-gate/)

