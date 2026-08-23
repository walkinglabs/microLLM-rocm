# Experiment 116：中位数很好看，尾部请求却等了三倍

## 问题

Experiment 115 的 8 条请求刚好平均填满两个 B4 桶。真实流量不会如此整齐。如果 6 条短请求
都只能进入 4-slot 小桶，另外 2 条短请求会排队，即使大桶仍有空 slot。

本轮固定三组反例：

```text
short-heavy  6 短 + 2 长，小桶排队
long-heavy   2 短 + 6 长，大桶排队
delayed      4 短在 step 0，4 长在 step 4 到达
```

每组比较统一 B8 和两个 B4 桶。总 slot、模型、权重、dtype、token 和输出长度不变。

## 为什么正式运行等了一天

第一次 GPU 选择为 `1% use / 0% VRAM`，但首个进程结束时外部任务已占 61% VRAM。post gate
立即停止，raw 为 0 行。第二次监控 GPU2/3 共 180 次，两卡显存始终 60%–90%，benchmark
没有启动。

最终 physical GPU 2 连续三次 `0/0` 后才运行。36 条记录 pre VRAM 最大 1%、post 最大 2%，
use 最大 5%，全部在固定门内。

## 正式结果

表中延迟只计算会排队或延迟到达的 focus requests。

### Qwen2.5-0.5B

| 流量 | TPS uniform→B4×2 | TTFT P50 | TTFT P95 | completion P50 | completion P95 |
|---|---:|---:|---:|---:|---:|
| short-heavy | 437.50→319.00 | 47.52→14.31 ms | 47.54→155.90 ms | 110.27→138.15 ms | 110.63→249.92 ms |
| long-heavy | 510.38→290.71 | 81.96→55.47 ms | 81.98→251.30 ms | 216.95→216.93 ms | 217.24→384.31 ms |
| delayed | 439.24→413.48 | 50.62→55.16 ms | 50.62→55.18 ms | 184.74→198.52 ms | 184.93→198.75 ms |

### DeepSeek Distill Qwen 1.5B

| 流量 | TPS uniform→B4×2 | TTFT P50 | TTFT P95 | completion P50 | completion P95 |
|---|---:|---:|---:|---:|---:|
| short-heavy | 256.39→186.78 | 84.90→24.76 ms | 84.91→266.81 ms | 190.18→234.38 ms | 190.55→425.54 ms |
| long-heavy | 298.00→170.32 | 145.25→99.21 ms | 145.26→431.53 ms | 373.97→370.66 ms | 374.33→655.36 ms |
| delayed | 255.06→239.99 | 95.38→102.56 ms | 95.39→102.57 ms | 321.02→343.50 ms | 321.22→343.72 ms |

![Traffic skew tail failure](../assets/traffic-skew-tail.svg)

## 中位数为什么会骗人

short-heavy 中，前 4 条短请求立即进入小桶，后 2 条等待。6 条请求的中位数主要由先进去的
4 条决定，因此 TTFT P50 改善约 70%；P95 却看到排队请求，Qwen/DeepSeek 分别变为统一池的
3.28×/3.14×。

long-heavy 更严格：长请求装不进小桶，6 条争抢 4 个大桶 slot。TTFT P95 变为 3.07×/2.97×，
吞吐只剩统一池的 57%。

delayed 流量中两个桶没有提供收益：吞吐下降约 5.9%，focus TTFT/completion 全面恶化约 7%–9%。

## 正确性

36/36 进程通过，六组 comparison 的 `differing_request_count=0`。分桶没有改变 token；失败是
调度公平性和资源利用率，不是模型数值错误。

## 决策

- 一个统一 B8 继续作为默认；
- 两个固定 B4 只保留为已知分布、显存受限时的显式策略；
- 拒绝根据 median TTFT 自动启用分桶；
- 下一候选必须让能装进大桶的短请求借用空 slot，并把 focus TTFT/completion P95 设为主门；
- 长请求无法借小桶，仍需要不同 slot 配方或 paged Cache，不能假装一次 work stealing 解决全部。

## 测量边界

arrival step 是逻辑调度步，不是固定 QPS wall-clock load generator。本实验足以证明固定桶的排队
反例，但不能代替生产请求到达过程。
