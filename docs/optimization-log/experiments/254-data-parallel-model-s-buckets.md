# Experiment 254 — Model-S给了我们几个真实bucket

Status: `natural multi-bucket baseline selected`

Model-S 15,586,176参数，B1T32，三种policy各三个轮换进程，每进程5 step，末步全参数审计。

| Limit | Buckets | Communication | Total | Peak/rank |
|---|---:|---:|---:|---:|
| 1 MiB | 45 | 9.18 ms | 21.76 ms | 549,089,280 B |
| 4 MiB | 12 | 15.49 ms | 28.29 ms | 549,089,280 B |
| 25 MiB | 3 | 6.825 ms | 19.76 ms | 603,383,808 B |

![Model-S data-parallel buckets](../assets/data-parallel-model-s-buckets.svg)

45个loss完全相同，9次末步rank参数差为0。三个bucket的25MiB是当前最佳reducer baseline，
相对4MiB省8.67ms，但每卡peak多54,294,528 bytes。

bucket count本身不能外推性能：45-bucket反而胜12-bucket，说明参数边界、pack/unpack、allocation
和collective shape共同决定结果。下一代码节点先给BucketStats补pack/unpack/temporary Tensor恒等式，
再设计gradient-as-bucket views与readiness，不直接写一个“异步”开关。

证据：[`Model-S bucket matrix`](../../../benchmarks/results/2026-08-25-data-parallel-model-s-bucket-matrix/)

