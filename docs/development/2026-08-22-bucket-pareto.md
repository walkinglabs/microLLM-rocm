# 2026-08-22：长度桶 Pareto 与 GPU 空闲门

## 起因

四个 B2 桶省 52.91% KV，但吞吐下降约 42%。本轮固定相同 8 请求，增加两个 B4 桶中间点。

## 被拒绝的第一轮

第一轮程序 18/18 成功，但 telemetry 在 17:23:53Z 出现 60% 外部 VRAM，随后达到 96%。
DeepSeek 四桶吞吐变成 131/141/157 tokens/s。它被标记为 `partial_invalid`，没有从中挑选看似
稳定的行。

因此 runner 增加：

```text
--physical-gpu-index 3
--max-idle-vram-percent 5
```

每个 fresh process 前后都读取物理卡状态。正式 18 条 pre VRAM 全为 0%，post 最大 2%。

## 正式结论

两个 B4 桶相对一个 B8 桶，在 Qwen/DeepSeek 上都得到：

```text
KV backing       -37.40%
throughput        -13.7% 到 -13.9%
median TTFT       -34.5% 到 -34.8%
TTFT p95          +6.9% 到 +7.5%
completion p50    +19.0% 到 +20.0%
engine peak       -3.1% 到 -3.6%
```

两个模型 1/2/4 桶 token exact。默认仍为一个 B8 桶；两个桶只是当前固定负载的 balanced opt-in。

## 下一步

当前 8 请求恰好均匀填满每个桶，不能代表真实在线流量。下一步先测：

- 6 条短请求、2 条长请求；
- 2 条短请求、6 条长请求；
- 请求分批延迟到达；
- 小桶排队而大桶空闲的 no-work-stealing 反例。

只有这些数据明确后，才选择 slot stealing、跨桶 decode 或 paged Cache。完整证据见
[Experiment 115](../optimization-log/experiments/115-bucket-pareto.md)。
