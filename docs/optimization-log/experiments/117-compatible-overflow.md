# Experiment 117：借两个大桶 slot，追回六成尾延迟

## 只修哪一个失败

Experiment 116 中，6 条短请求争抢 4 个小桶 slot，而大桶只放了 2 条长请求。候选只做一件事：

```text
最小兼容桶负载 < slots  → 留在最小桶
最小兼容桶已满          → 找第一个有即时容量的更大兼容桶
所有兼容桶已满          → 回最小桶排队
```

请求提交后不迁移。长请求不能进入容量不足的小桶，因此 long-heavy 必须保持反例。

## 第一次为什么被拒绝

GPU 环境完全干净，但 runner 在第一条 overflow case 报 `invalid bucket routing`。代码错误地计算：

```text
load = active_request_count + pending_request_count
```

`active_request_count` 已包含 pending，所以 4-slot 小桶在只有 2 条请求时就被误判满。第一次只保留
6 条 uniform/fixed raw，没有 overflow summary。

修复后新增精确门：前 4 条留在小桶，第 5 条才溢出。官方 Qwen smoke 路由为
`[0,0,0,0,1,1,1,1]`、overflow count 2，随后从新目录重跑全部 54 进程。

## 正式结果

U/F/O 分别表示 uniform、fixed two-bucket、compatible overflow。

### Qwen2.5-0.5B

| 流量 | TPS U/F/O | focus TTFT P95 U/F/O | completion P95 U/F/O |
|---|---:|---:|---:|
| short-heavy | 437.36 / 319.45 / 361.96 | 47.57 / 154.79 / 58.56 ms | 110.57 / 248.70 / 149.07 ms |
| long-heavy | 507.23 / 291.12 / 290.27 | 86.37 / 250.63 / 250.58 ms | 222.48 / 383.12 / 383.27 ms |
| delayed | 439.49 / 416.52 / 416.62 | 50.88 / 55.70 / 55.65 ms | 185.14 / 198.34 / 198.55 ms |

### DeepSeek Distill Qwen 1.5B

| 流量 | TPS U/F/O | focus TTFT P95 U/F/O | completion P95 U/F/O |
|---|---:|---:|---:|
| short-heavy | 256.01 / 187.61 / 211.73 | 85.36 / 266.50 / 104.58 ms | 190.90 / 425.85 / 256.97 ms |
| long-heavy | 298.94 / 170.57 / 170.40 | 144.70 / 430.22 / 432.22 ms | 374.61 / 653.33 / 655.81 ms |
| delayed | 254.56 / 240.58 / 239.96 | 95.00 / 103.10 / 102.89 ms | 321.34 / 343.90 / 344.25 ms |

![Compatible overflow result](../assets/compatible-overflow.svg)

## 候选修复了多少

short-heavy 相对固定桶：

```text
吞吐                 +12.9% 到 +13.3%
focus TTFT P95       -60.8% 到 -62.2%
focus completion P95 -39.7% 到 -40.1%
```

long-heavy 和 delayed 没有请求溢出，candidate/fixed 各主指标都在约 0.5% 内，说明未需要借槽时
没有实质扰动。

## 为什么仍不设默认

short-heavy 相对统一池仍然：

```text
吞吐                 约 -17.3%
focus TTFT P95       约 +23%
focus completion P95 约 +35%
```

overflow 修复了固定桶内部的不均衡，但没有恢复跨桶 B8 batching。默认继续是 uniform；overflow
只作为显式的两桶增强策略。

## 正确性和环境

- 54/54 fresh process 通过；
- 六组对 fixed、uniform 都 token exact；
- short-heavy 两模型 actual route 都是 `[0,0,0,0,1,1,1,1]`，count 2；
- long/delayed count 0 且 candidate route 等于 fixed；
- pre VRAM/use 最大 1%/2%，post 最大 3%/5%；
- Release 319/319，sanitizer 215/215。

## 下一步

长请求无法借小桶，固定 slot 比例仍会在 long-heavy 下损失约 43% 吞吐和约 3× TTFT P95。
下一步不继续堆 admission 规则，而应比较：

1. 根据已知流量改变 2:6/4:4/6:2 slot 配方；
2. 动态/paged Cache 是否能让 slot 不再绑定固定 capacity；
3. 跨桶 decode 是否值得为约 17% 的剩余 short-heavy 吞吐差增加布局复杂度。
