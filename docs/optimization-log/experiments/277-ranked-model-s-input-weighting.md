# Experiment 277 — Model-S的B1/B2能否等价CPU B3

Status: `synchronous Model-S token weighting kept`

固定Model-S T32，两rank rows `[1,2]`。local tokens 32/64，average 48，scale
`0.666666687/1.333333373`。

| Gate | Result |
|---|---:|
| equal-only | 2/2 rank拒绝 |
| rank Max/RMS | 0 / 0 |
| CPU parameter Max/RMS | 0.007760 / 3.639e-6 |
| weighted loss diff | 3.20e-7 |
| compared | 57 Tensor / 15,586,176 values |
| engine peak | 275,790,348 bytes |

![Ranked Model-S input weighting](../assets/ranked-model-s-input-weighting.svg)

同步weighted模式在大参数路径保持global-batch语义。一步时间包含进程/库首次初始化，只作smoke，
不作性能比较。

ready overlap仍未支持weighted：当前hook可能在统一post-backward scale之前enqueue。下一节点把
local scale移动到每个leaf ready hook、Event record之前，再比较同步weighted与weighted overlap。

证据：[`ranked Model-S input weighting`](../../../benchmarks/results/2026-08-25-ranked-model-s-input-weighting/)
