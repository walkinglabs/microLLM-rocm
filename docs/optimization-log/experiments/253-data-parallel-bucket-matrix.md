# Experiment 253 — tiny模型能不能研究bucket overlap

Status: `matrix complete; tiny overlap workload rejected`

四种bucket限制各三个轮换顺序进程，每进程20 step，只在末步做参数审计。

| Limit | Buckets | Communication | Total |
|---|---:|---:|---:|
| 4 B | 12 | 1.26 ms | 2.98 ms |
| 64 B | 12 | 1.18 ms | 2.75 ms |
| 4 KiB | 1 | 0.34 ms | 2.03 ms |
| 4 MiB | 1 | 0.39 ms | 2.16 ms |

![Data parallel bucket matrix](../assets/data-parallel-bucket-matrix.svg)

240个loss逐项相同，12个进程末步参数差均为0。4KiB与4MiB都生成同一个bucket，执行图相同，
时间差是进程噪声。12个tiny bucket把通信放大约3–4倍并拖慢total。

因此tiny workload不能研究有意义的overlap：单bucket没有分阶段机会，人为切12个小bucket又被
collective/pack开销支配。下一步必须给CLI增加Model-S多bucket workload，再决定persistent
bucket与readiness state machine。

证据：[`bucket matrix`](../../../benchmarks/results/2026-08-25-data-parallel-bucket-matrix/)

