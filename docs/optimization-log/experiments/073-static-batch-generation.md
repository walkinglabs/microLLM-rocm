# Experiment 073 — 第一次真正把多条请求放进同一次forward

`generate_batch()`接收多个prompt row。内容可以不同，但长度、`max_new_tokens`和采样配置
必须相同。它使用一个`KVCache(batch=B)`：full prefill走`[B,T]`，每步decode走`[B,1]`，
greedy选择用GPU row-wise argmax。

CPU测试同时覆盖greedy和随机top-k，每一行与相同seed的独立`generate()`一致；HIP三条不同
prompt与CPU一致。空batch、不等长prompt、context超限和错误逐层dtype全部显式拒绝。

## 性能

与Experiment 072相同的106,816参数模型；兼容请求统一prompt length 8、生成4 token。
CPU/HIP、B1/2/4/8、1 warm-up、5 measured、3个新进程：

| device | B | reference tok/s | static batch tok/s | speedup | scaling efficiency |
|---|---:|---:|---:|---:|---:|
| CPU | 1 | 332 | 331 | 0.997× | 100.0% |
| CPU | 2 | 327 | 494 | 1.481× | 74.5% |
| CPU | 4 | 328 | 647 | 1.979× | 48.8% |
| CPU | 8 | 330 | 752 | 2.255× | 28.4% |
| HIP | 1 | 334 | 337 | 1.008× | 100.0% |
| HIP | 2 | 332 | 654 | 1.958× | 97.1% |
| HIP | 4 | 334 | 1,256 | 3.762× | 93.3% |
| HIP | 8 | 334 | 2,443 | 7.306× | 90.7% |

![Static batch generation](../assets/static-batch-generation.svg)

24/24进程输出一致。B8 Cache为49,152B，随batch线性增长；engine peak从B1 1.306MB增到
B8 1.479MB。

## 决定

保留`generate_batch()`和benchmark。它是第一个真实跨请求batch primitive，但不是continuous
batching：不能容纳晚到请求、不同prompt长度、不同完成时间或slot refill。下一节点要把
ReferenceScheduler的生命周期与这个batched primitive连接，而不是再写另一套答案语义。
