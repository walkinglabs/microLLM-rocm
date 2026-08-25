# Experiment 259 — pack copy归零，为什么整步反而更慢

Status: `discard model route`

同一二进制轮换transient、bucket-view和direct accumulation。全部使用Model-S B1T32、
25MiB/3bucket、5step和末步参数审计，每策略三个进程，steady聚合step 2–5。

| Policy | Forward/backward | Comm | Total | Peak |
|---|---:|---:|---:|---:|
| transient | 12.105 ms | 6.740 ms | 19.630 ms | 603,383,808 |
| bucket views | 10.400 ms | 3.585 ms | 14.900 ms | 636,652,808 |
| direct | 12.535 ms | 1.650 ms | 15.035 ms | 623,447,040 |

![Direct bucket-gradient discard](../assets/data-parallel-direct-bucket-gradient-discard.svg)

Direct路径的114个pack/unpack copy全部为0，communication相对view快2.173×，peak少
13,205,768B；但forward/backward只有0.830×，total只有0.991×，没有过1.01门。45个loss和
9次末步rank参数完全一致，所以这是性能反例，不是数值失败。

原因：当前backward算子仍先申请并写出普通gradient，leaf accumulation再launch add写入目标
view。它删除通信copy，却没有让producer直接写最终地址。模型C++/CLI route撤回；独立验证的
leaf accumulation target保留，只有具体producer out-kernel能同时删除临时输出与add时才重开。

证据：[`direct bucket gradient matrix`](../../../benchmarks/results/2026-08-25-data-parallel-direct-bucket-gradients/)
