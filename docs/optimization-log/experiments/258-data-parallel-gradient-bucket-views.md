# Experiment 258 — all-reduce完了为什么还要把gradient复制出来

Status: `kept explicit, not default`

同一二进制轮换transient、persistent-copy和bucket-view。全部使用Model-S B1T32、25MiB/
3bucket、5step和末步参数审计，每策略三个进程，steady只聚合step 2–5。

| Policy | Unpack Storage/copy | Comm | Total | Live | Peak |
|---|---:|---:|---:|---:|---:|
| transient | 114 / 114 | 6.925 ms | 20.345 ms | 498,757,632 | 603,383,808 |
| persistent-copy | 114 / 114 | 4.045 ms | 15.880 ms | 623,447,040 | 761,342,216 |
| bucket views | 0 / 0 | 3.575 ms | 14.885 ms | 498,757,632 | 636,652,808 |

![Gradient-as-bucket views](../assets/data-parallel-gradient-bucket-views.svg)

view相对copy的communication/total为1.131×/1.067×，相对transient为1.937×/1.367×；
45个loss与9次末步rank参数完全一致。相对copy的live和peak各省124,689,408B，live已回到
transient基线。

但peak仍比transient多33,269,000B：backward先产生新parameter gradients，而persistent
bucket仍存活。因此保持显式。下一步在backward前把bucket views设为gradient累加目标，删除
114次pack copy与双表示峰值；在该门通过前不宣称ready-overlap或默认persistent reducer。

证据：[`gradient view matrix`](../../../benchmarks/results/2026-08-25-data-parallel-gradient-views/)
