# Experiment 256 — all-reduce后为什么还要再申请一份bucket

Status: `kept as default`

同一二进制只切换average是否原地。Model-S B1T32、25MiB/3bucket、5step、末步参数审计，
每种三个交替顺序进程。

| Policy | Average Tensor | Backend alloc | Temporary bytes | Comm | Total |
|---|---:|---:|---:|---:|---:|
| allocating | 6 | 126 | 374,068,224 | 6.60 ms | 19.21 ms |
| in-place | 0 | 120 | 249,378,816 | 5.20 ms | 17.35 ms |

![Data parallel in-place average](../assets/data-parallel-inplace-average.svg)

communication 1.269×，total 1.107×，peak不变；30个loss逐项相同，末步rank参数差为0，
RCCL 22/22通过。

in-place保留为默认，并保留显式allocating control用于回归。它删除一整份rank-local bucket表示
并稳定bucket地址，是persistent plan前提。但仍有120次backend allocation和228次copy。

证据：[`in-place matrix`](../../../benchmarks/results/2026-08-25-data-parallel-inplace-average/)

