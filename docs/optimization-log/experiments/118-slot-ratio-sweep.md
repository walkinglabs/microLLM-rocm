# Experiment 118：短流量选 6:2，长流量选 2:6

## 问题

Experiment 117 让短请求借用大桶，修复了固定 4:4 的部分尾延迟。但如果流量分布已知，直接把
slot 配给正确容量，是否比 overflow 更简单、更有效？

固定三种配方：

```text
小桶:大桶 = 2:6、4:4、6:2
小桶 capacity 520，大桶 capacity 2064
总 slot 始终 8
```

每种都与统一 B8 比较。请求、权重、token、dtype 和输出长度不变。

## 正式结果

### Qwen2.5-0.5B

| 流量 | 配方 | TPS | KV MiB | focus TTFT P95 | completion P95 |
|---|---:|---:|---:|---:|---:|
| short-heavy | uniform | 439.60 | 193.50 | 47.55 ms | 110.43 ms |
| short-heavy | 2:6 | 283.10 | 157.31 | 250.77 ms | 282.38 ms |
| short-heavy | 4:4 | 320.32 | 121.13 | 154.54 ms | 248.66 ms |
| short-heavy | 6:2 | 371.37 | 84.94 | 19.23 ms | 143.59 ms |
| long-heavy | uniform | 511.00 | 193.50 | 81.28 ms | 216.33 ms |
| long-heavy | 2:6 | 447.29 | 157.31 | 87.04 ms | 250.26 ms |
| long-heavy | 4:4 | 290.54 | 121.13 | 251.11 ms | 384.06 ms |
| long-heavy | 6:2 | 233.41 | 84.94 | 345.74 ms | 478.48 ms |

### DeepSeek Distill Qwen 1.5B

| 流量 | 配方 | TPS | KV MiB | focus TTFT P95 | completion P95 |
|---|---:|---:|---:|---:|---:|
| short-heavy | uniform | 256.69 | 451.50 | 85.29 ms | 191.01 ms |
| short-heavy | 2:6 | 168.29 | 367.06 | 420.53 ms | 474.39 ms |
| short-heavy | 4:4 | 186.80 | 282.63 | 267.20 ms | 426.20 ms |
| short-heavy | 6:2 | 219.28 | 198.19 | 34.99 ms | 244.98 ms |
| long-heavy | uniform | 298.58 | 451.50 | 143.85 ms | 374.15 ms |
| long-heavy | 2:6 | 260.58 | 367.06 | 152.69 ms | 428.72 ms |
| long-heavy | 4:4 | 170.29 | 282.63 | 431.83 ms | 655.60 ms |
| long-heavy | 6:2 | 136.89 | 198.19 | 592.71 ms | 815.75 ms |

![Slot ratio sweep](../assets/slot-ratio-sweep.svg)

## 最佳静态点

short-heavy 的 6:2：

```text
吞吐保留           84%–85%
KV backing          约为uniform的44%
TTFT P95            约为uniform的40%–41%
completion P95      约为uniform的128%–130%
```

long-heavy 的 2:6：

```text
吞吐保留           约87%
KV backing          约为uniform的81%
TTFT P95            约为uniform的106%–107%
completion P95      约为uniform的115%–116%
```

48/48 进程通过，四个 sweep 在三种 ratio 下都与 uniform token exact。

## 关键反例

静态最佳会随流量完全翻转：short-heavy 要 6:2，long-heavy 要 2:6。错误配方会让 TTFT P95
变成 uniform 的 3×–5×。因此不能把某个 ratio 按模型名硬编码成“优化”。

## 决策

- 允许调用方显式配置 slot ratio；
- uniform 继续默认；
- 不增加自动模型名规则；
- 如果服务方已知并稳定控制长度分布，可选 workload-matched ratio；
- 流量随时间变化时，真正的问题已不是寻找第四个静态比例，而是如何让 capacity 动态分配。

下一节点应先做一个小型动态容量/重建策略或 paged Cache 设计实验，并把配方切换成本、allocator
保留显存和正在运行请求的生命周期写进合同。
