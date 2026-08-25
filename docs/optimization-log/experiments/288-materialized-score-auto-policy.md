# Experiment 288：不传开关时，默认策略真的生效了吗

Status: scoped auto policy retained

## 因果对照

current每个进程显式传`materialized=false`；candidate完全不传开关，并必须在JSON报告
`auto-enabled`、eligible=true。Qwen/DeepSeek、T2048、B1/B2，每格三对fresh process、N32。

![Automatic policy matrix](../../../benchmarks/results/2026-08-25-materialized-attention-auto-matrix/matrix.svg)

| 模型 | B | explicit-off tok/s | auto tok/s | speedup | logits |
|---|---:|---:|---:|---:|---|
| Qwen | 1 | 112.79 | 133.38 | 1.1836x | 位级相同 |
| Qwen | 2 | 224.90 | 264.90 | 1.1777x | 位级相同 |
| DeepSeek | 1 | 67.22 | 91.94 | 1.3687x | 位级相同 |
| DeepSeek | 2 | 134.05 | 177.74 | 1.3259x | 位级相同 |

四格三组leave-one全部过门，token相同、peak delta为0。每格N32多32次backend allocation，保留为
下一workspace候选的证据，不在本实验混入第二项改动。

## 决定

保留有界auto：gfx942、BF16 KV、uniform cached、已测head签名、prefix>=2048。显式on/off优先。
这不是所有AMD GPU或所有模型的默认声明。Step 105到此完成，Step 106重新profile新的默认路径。

证据：[`automatic policy matrix`](../../../benchmarks/results/2026-08-25-materialized-attention-auto-matrix/)
