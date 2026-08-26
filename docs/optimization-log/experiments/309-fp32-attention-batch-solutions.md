# Experiment 309：Exact候选很多，但没有一个四个batch都不回退

Status: operator performance rejected; model counterfactual selected

## 真实descriptor

QK固定`M2048 N2048 K128`、P×V固定`M2048 N128 K2048`，请求B1/2/4/8只把backend batch count
改成12/24/48/96。B1的12-head输入块在GPU上位级复制，避免把不同输入误叫算法不稳定。

每个shape查询64个solution，先取共同index。候选必须通过CPU sentinel、完整B1/default、所有请求块
位级一致，才做1 warm-up + 3 Event/wall。

## 数值结果

- QK：共同34，correctness 34，跨batch exact 34；default相同输入Max `2.98e-7`。
- P×V：共同2，correctness 2，跨batch exact 2；default相同输入Max `1.19e-7`。

所以“没有可保序solution”被推翻。相同index确实可以让两类算子跨batch位级相同。

## 性能反例

没有候选满足四个batch最差speedup ≥0.95：

| Operation | best exact | B1 | B2 | B4 | B8 | min | geometric mean | admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QK | 304681 | 0.933× | 1.111× | 0.916× | 1.083× | 0.916× | 1.007× | -1 |
| P×V | 295716 | 0.535× | 1.003× | 1.360× | 1.026× | 0.535× | 0.930× | -1 |

![Attention solutions](../../../benchmarks/results/2026-08-26-fp32-attention-batch-invariance/attention-solutions.svg)

## 决定

两类operator默认都不改变。304681/295716只进入一次完整模型反驳：局部P×V B1回退46.5%可能在整网
被稀释，而完整logits跨batch可能改善；这两件事都必须实测。下一节点增加QK/PV独立scope，避免solution
误命中projection或其他同形GEMM，然后跑DeepSeek B1/2/4/8 fresh-process logits/吞吐/peak。

证据：[`result directory`](../../../benchmarks/results/2026-08-26-fp32-attention-batch-invariance/)
