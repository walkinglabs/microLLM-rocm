# Experiment 074 — 请求先分组，再进入static batch

`AdmissionBatchScheduler`保存当前等待请求，`drain()`按精确compatibility key稳定分组：prompt
长度、生成长度、temperature、top-k、seed、Cache dtype/逐层策略都相同才进入一组。
每组调用一次`generate_batch()`，不兼容请求走B1 singleton。

CPU测试覆盖一个B2组和两个singleton，再在下一次drain接收晚到B2组；每个结果与独立
`generate()`一致。HIP测试形成B3+singleton并与CPU逐row对齐。metrics记录drain、group、
singleton、batched request和最大batch。

## 性能

固定tiny模型将请求按每4条形成兼容组。CPU/HIP、1/2/4/8/16请求、三进程中位数：

| device | requests | admission tok/s | vs serial | groups | max B |
|---|---:|---:|---:|---:|---:|
| CPU | 1 | 371 | 0.999× | 1 | 1 |
| CPU | 2 | 519 | 1.390× | 1 | 2 |
| CPU | 4 | 687 | 1.816× | 1 | 4 |
| CPU | 8 | 638 | 1.835× | 2 | 4 |
| CPU | 16 | 629 | 1.773× | 4 | 4 |
| HIP | 1 | 336 | 1.009× | 1 | 1 |
| HIP | 2 | 655 | 1.965× | 1 | 2 |
| HIP | 4 | 1,260 | 3.780× | 1 | 4 |
| HIP | 8 | 1,253 | 3.768× | 2 | 4 |
| HIP | 16 | 1,259 | 3.779× | 4 | 4 |

![Admission batch scheduler](../assets/admission-batch-scheduler.svg)

30/30进程输出一致。HIP在B4达到约1260 tok/s，继续增加队列只增加串行group数，吞吐不再
增长。这个平台期是正确架构边界，不应通过“16请求总共生成更多token”伪装扩展。

## 决定

保留admission scheduler与metrics。它支持跨drain到达和兼容分组，但每个group仍一次性生成
完成，不能在token级补slot。下一节点需要活跃slot refill；若做不到可变position/RoPE，必须
明确保持当前平台，而不能称continuous batching。
