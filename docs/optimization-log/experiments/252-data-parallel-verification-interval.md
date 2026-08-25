# Experiment 252 — 每步全参数检查应该算进训练吗

Status: `kept; default interval remains 1`

三种policy各三个新进程，顺序轮换。每进程20 step，第1步lazy setup保留在raw，steady只聚合
step 2–20。

| Policy | Checks | Total | Verification | Speedup |
|---|---:|---:|---:|---:|
| every-step | 20 | 2.96 ms | 0.395 ms | 1.000× |
| final-step | 1 | 2.38 ms | 检查时0.370 ms | 1.244× |
| disabled | 0 | 2.52 ms | 0 | 1.175× |

![Data parallel verification interval](../assets/data-parallel-verification-interval.svg)

180个loss逐项相同，检查次数严格20/1/0。final-step比disabled略快是tiny workload噪声，
不能解释成检查会加速；可靠结论是每步host审计确实污染hot path。

实现还修复了隐式同步：旧的to_vector审计顺便等待optimizer。现在optimizer显式同步并计入
optimizer_ms，跳过审计不改变step完成语义。默认interval仍为1；0/N必须显式选择。

下一步清理bucket测量：用final-step审计保留末步参数证据，扫描多bucket大小，不能在单bucket
tiny模型上宣称overlap。

证据：[`verification matrix`](../../../benchmarks/results/2026-08-25-data-parallel-verification-matrix/)

