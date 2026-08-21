# Experiment 103 — 固定请求后，1/2/4/8 slot才是公平比较

Experiment 102的S2/S4请求集不同，不能直接计算并行效率。本节点固定同一批8条short和8条long
请求，只改变slot数，并保留失败前后的完整48进程矩阵。

## 第一轮：30 pass，18 stable fail

两个模型完全相同的6个model/group/slot组合稳定失败：short S1、long S1、long S2；每个组合
三进程，共18条。没有OOM或timeout。错误都来自全slot回收后的prefill：row position已经归零，
但可复用Storage仍defined，错误进入只允许空Storage的首次prefill。

最小修复让全row fast path同时要求“所有layer Storage都未创建”。单slot不同长度refill的CPU/HIP
回归通过后，原矩阵48/48执行通过。

![Fixed-request slot sweep](../assets/continuous-slot-sweep.svg)

## 三进程中位速度和S1效率

| 模型/组 | S1 tok/s | S2 speedup/eff | S4 speedup/eff | S8 speedup/eff |
|---|---:|---:|---:|---:|
| Qwen short | 296.05 | 1.469× / 73.5% | 2.568× / 64.2% | 4.323× / 54.0% |
| Qwen long | 149.76 | 1.831× / 91.5% | 2.771× / 69.3% | 3.216× / 40.2% |
| DeepSeek short | 180.03 | 1.589× / 79.5% | 2.664× / 66.6% | 4.688× / 58.6% |
| DeepSeek long | 87.93 | 1.822× / 91.1% | 2.723× / 68.1% | 3.137× / 39.2% |

短请求S8仍有54%–59%效率。长请求S8只有约39%–40%；从S4到S8，吞吐只提高15%–16%。

## Cache和精度代价

KV allocated严格随slot线性增长。long S8相对S4从96.75→193.50 MiB（Qwen）、
225.75→451.50 MiB（DeepSeek），但byte utilization都从75.15%降到46.85%。增加slot并没有让
每条预留的最大长度同时变成有效内容。

Qwen short/long与DeepSeek long跨slot完整token一致。DeepSeek short的S1/S2 checksum相同，S4/S8
形成另一组；唯一分叉请求是index 5，首个分叉为生成位置4。runner不再因这个失败丢弃summary，
而是写`execution_status=pass`和总状态`complete_with_recorded_accuracy_failures`。

## 结论

refill生命周期bug被同一48进程矩阵反驳，修复保留。公平batch证据说明S8对短请求仍有价值，长
请求则要用近2倍KV换约15%额外吞吐。下一节点不是继续加slot，而是记录DeepSeek首个分叉位置的
两条路径logits/top-2 margin，并研究按请求长度分配Cache块。

数据见[`103-data`](103-data/)。
