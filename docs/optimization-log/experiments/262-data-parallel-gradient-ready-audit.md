# Experiment 262 — 自然bucket真的会在backward结束前ready吗

Status: `admit Event-based overlap prototype`

Model-S B1T32、25MiB bucket，三个fresh process各3step、两rank。hook只在leaf最后一条贡献
完成入队后记录；reducer仍完全同步。

9个step的57参数permutation完全相同，两rank逐项一致，恰为parameter order逆序；3次末步参数
差均为0。

| Bucket | 参数范围 | Bytes | 完成位置 | backward前完成 |
|---|---:|---:|---:|---:|
| 0 | 0–21 | 26,156,544 | 57/57 | no |
| 1 | 22–55 | 23,605,248 | 35/57 | yes |
| 2 | 56 | 12,582,912 | 1/57 | yes |

![Gradient-ready bucket order](../assets/data-parallel-gradient-ready-order.svg)

两个自然bucket存在结构性窗口，因此准入Event+async all-reduce原型。这里没有通信时间线和
speedup，不能写成“已经overlap”；下一节点必须保留同步control、loss/参数/peak门。

证据：[`gradient-ready audit`](../../../benchmarks/results/2026-08-25-data-parallel-gradient-ready-audit/)
