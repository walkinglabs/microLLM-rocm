# Experiment 217 — 三道作业能不能一次交

Status: `discard before model route`

## 共享关系是真的

Q/K/V 的 weight gradient 都是：

```text
inputᵀ @ output_gradient
```

gate/up 也是同一形式。它们共享 `inputᵀ`，所以数学上适合 grouped GEMM。Experiment 216 的
profile 又说明 GEMM 已占 59%/64%，值得先做能力探针。

## 两条路径都没有可用算法

我们测试：

1. 不复制 input，使用 grouped `N,T`；
2. 只转置一次共享 input，使用 grouped `N,N`，并要求未来计时包含转置。

| Layout | Algorithm inventory | Official cases | Supported |
|---|---:|---:|---:|
| direct `N,T` | 8,153 | 4 | 0 |
| materialized `N,N` | 9,172 | 4 | 0 |

四个 case 是 Qwen/DeepSeek × QKV/gate-up，全部为 rows=512 FP32 weight gradient。即使
gate/up 两个 GEMM 宽度完全相同也没有候选，所以不能把失败归因于 QKV 宽度不同。

![Grouped weight-gradient capability](../assets/grouped-weight-gradient-discard.svg)

## 决定

不修改 Autograd，不加入协调多个输出 gradient 的脆弱计数器，也不拿三个普通 GEMM fallback
冒充 grouped 支持。保留独立 benchmark、8-case runner和schema测试。

下一条可反驳路线是显式打包 output gradient/weight，通过一个普通大 GEMM 计算 packed result；
它必须把 pack/split 与 input-gradient 求和全部计入，并先证明局部上限足够。

原始数据在
[`benchmarks/results/2026-08-24-grouped-weight-gradient-discard/`](../../../benchmarks/results/2026-08-24-grouped-weight-gradient-discard/)。
