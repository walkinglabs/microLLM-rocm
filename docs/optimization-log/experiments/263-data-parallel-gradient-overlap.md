# Experiment 263 — 两个早ready bucket能隐藏多少通信

Status: `kept explicit, not default`

同一二进制轮换transient、同步persistent views、Event overlap views。Model-S B1T32、25MiB/
3bucket、5step、末步参数审计，每策略3进程，steady取step 2–5。

| Policy | Finish/communication | Total | Peak |
|---|---:|---:|---:|
| transient | 6.815 ms | 20.435 ms | 603,383,808 |
| synchronous views | 3.560 ms | 15.025 ms | 636,652,808 |
| overlap views | 1.550 ms | 14.790 ms | 636,652,808 |

![Gradient-ready overlap](../assets/data-parallel-gradient-overlap.svg)

overlap相对sync view total 1.0159×，finish wait缩短2.297×，peak不变；45个loss和9次参数门
完全一致，12个later step均enqueue 3 bucket且0通信allocation。相对transient total 1.3817×，
但peak仍多33,269,000B。

因此保留显式能力，不设默认。单进程rank0→rank1 backward限制了窗口；下一节点转
one-process-per-GPU communicator/init/error合同，再谈标准DDP overlap和默认。

证据：[`overlap matrix`](../../../benchmarks/results/2026-08-25-data-parallel-gradient-overlap/)
