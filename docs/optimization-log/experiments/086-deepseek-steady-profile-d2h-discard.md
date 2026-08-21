# Experiment 086 — 少21次D2H为什么反而让B8慢了14%？

Experiment 085把剩余性能缺口缩到DeepSeek T2048：Release steady decode在B1/B8只有PyTorch的
0.866×/0.671×。这次先profile，不先猜Kernel。

## 冻结profile合同

```text
模型        DeepSeek Distill Qwen 1.5B
shape       T2048，B1/B8
路径        BF16 KV、steady decode N8
binary      Experiment 085冻结Release/gfx942
设备        MI300X VF / GPU1
trace       rocprofv3 kernel、HIP API、allocation、copy
```

每条trace包含一次warm-up和一次measured Cache prefill，以及各8次warm-up/measured decode。
因此GEMM总表混有prefill，不能把所有GEMM时间都算成decode；`cached_attention_fused_kernel`
只出现在decode，可以直接归因。

## 第一热点是Cache读取，不是Cache写入

| 指标 | B1 | B8 |
|---|---:|---:|
| 全部Kernel | 343.62 ms | 767.81 ms |
| cached Attention | 158.86 ms / 448 calls | 167.87 ms / 448 calls |
| 单次cached Attention | 354.60 us | 374.71 us |
| GEMM（含两次prefill） | 103.70 ms | 376.84 ms |
| argmax | 4.68 ms | 4.81 ms |
| KV store | 1.51 ms | 1.62 ms |

448次正好是`28层 × 8 token × (warm-up + measured)`。假设两轮成本接近，cached Attention
约占measured decode wall的62.1%/60.5%。B8同时处理8行，它的Attention总时间只比B1高
5.7%，说明batch并行有效，但每个token都要重新读取T2048的K/V并做规约。

B8还有第二个系统热点：`hipFree`从B1的231.73 ms升到677.82 ms。此前正式三步矩阵中，
B8 backend allocation曾达到13,884，而cache reuse只有2,442，说明exact-size pool对分配顺序
十分敏感。

## 最小候选：token最后一次性回host

基线每个measured token都会把device argmax结果带回host：

```text
N8 × measured 3次 = 24次D2H
```

候选增加caller-owned argmax输出和GPU token history，公共greedy generation与benchmark都在
结束时只做一次D2H。候选单测证明token逐项一致，D2H calls从24降到3，字节数不变。

## 三对交替Release结果

| Batch | baseline tok/s（三次） | candidate tok/s（三次） | median速度比 |
|---:|---|---|---:|
| 1 | 66.85 / 66.59 / 66.73 | 66.92 / 66.88 / 66.88 | 1.002× |
| 8 | 495.58 / 509.58 / 504.72 | 503.35 / 433.76 / 434.76 | 0.861× |

B1中性，但B8明确越过5%拒绝门。更重要的是机制可解释：

| B8中位数counter | baseline | candidate |
|---|---:|---:|
| D2H calls | 24 | 3 |
| backend allocations | 874 | 13,863 |
| cache reuse | 15,452 | 2,442 |
| peak | 6.93 GiB | 6.93 GiB |

小history改变了分配/释放的相位，恰好触发allocator cache miss风暴。少同步是真的，整体变慢也
是真的；不能拿B1的绿色结果覆盖B8失败。

## 决定

候选全部回退，当前源码仍是Experiment 085路径。profile和失败数据保留：

1. `cached_attention_fused_kernel`是两种batch共同的设备第一热点；
2. B8 allocator相位敏感是阻止其他小优化稳定生效的第二热点；
3. argmax Kernel与KV store都不是优先目标；
4. token D2H可在allocator稳定后重试，但不能现在合入。

下一节点Experiment 087只处理allocator/cache reuse，不同时改Attention数学或token路径。

![DeepSeek steady profile and rejected D2H candidate](../assets/deepseek-steady-profile-d2h-discard.svg)

数据见[`086-data`](086-data/)。
