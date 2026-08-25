# Experiment 257 — 每一步都重新申请bucket真的必要吗

Status: `kept explicit, not default`

同一二进制只切换persistent gradient bucket。Model-S B1T32、25MiB/3bucket、in-place
average、5step、末步参数审计，每种三个交替顺序进程；steady只聚合step 2–5。

| Policy | 后续backend alloc | Comm | Total | Live bytes | Peak bytes |
|---|---:|---:|---:|---:|---:|
| transient | 120 | 7.070 ms | 21.025 ms | 498,757,632 | 603,383,808 |
| persistent | 0 | 4.205 ms | 16.360 ms | 623,447,040 | 761,342,216 |

![Persistent data-parallel buckets](../assets/data-parallel-persistent-buckets.svg)

通信提高1.681×，total提高1.285×；30个loss逐项相同，6次末步rank参数检查差为0，12个
persistent后续step的communication allocation/backend/cache全部为0。

但第一版同时长期持有6个bucket和114个unpacked Tensor，live增加124,689,408B，peak增加
157,958,408B。因此保留显式能力和地址稳定合同，不设为默认。下一反驳实验让参数gradient直接
成为reduced bucket的连续view，目标是删掉114个unpacked Storage和114次unpack copy。

证据：[`persistent bucket matrix`](../../../benchmarks/results/2026-08-25-data-parallel-persistent-buckets/)
