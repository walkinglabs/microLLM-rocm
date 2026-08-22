# 2026-08-22：共享权重的长度分桶 KV Cache

## 当天问题

固定 8 请求的长上下文矩阵显示，统一 S8 Cache 只有 46.85% byte utilization。短 row 也按最长
2064-token row 预留 backing，浪费是结构性的，不是 allocator 统计误差。

## 任务合同

只增加一个可选的长度分桶 policy：

- 所有桶共享同一个 `TransformerModel`；
- 每个桶复用现有 `ContinuousBatchScheduler` 并拥有独立 KV Cache；
- 请求按 `prompt + max_new_tokens` 进入最小兼容桶；
- 外层 ID 不与各子调度器本地 ID 混淆；
- 未提供桶时，旧接口和默认统一池不改变；
- 第一版不增加 slot stealing、多 Stream、动态桶或 paged Cache。

## 实现和审查

新增公共 `LengthBucketedBatchScheduler`、`--continuous-cache-buckets` 和 JSON 路由证据。审查中
专门拒绝了“把每个桶都分配 8 个 slot”的假节省方案；总 slot 必须仍为 8，物理公式是：

```text
sum(bucket capacity × bucket slots × per-token KV bytes)
```

峰值 active Cache 也不能把各桶历史峰值简单相加；外层在每次子调度完成后采样所有桶的瞬时和。

## 测试

- CPU：最小桶路由、全局 ID、排队而不偷 slot、取消、错误配方和独立生成对齐；
- HIP：BF16 Cache 的 CPU/HIP token、完成原因、理论 bytes 和 transfer 合同；
- CLI：官方 Qwen 权重真实加载、桶配置 `[10:1,34:1]` 与路由 `[0,1]`；
- ASan/UBSan：214/214；
- Release 全套：318/318，2 项按构建条件跳过；
- Python matrix/CLI 合同：11/11。

## 正式 MI300X 结果

四桶 `264×2,520×2,1040×2,2064×2` 与统一 S8 各跑 3 个 fresh process。Qwen 与
DeepSeek 共 12/12 通过，两模型跨 policy token exact。

```text
KV backing        -52.91%
median TTFT        -56% 到 -57%
throughput         -41.9% 到 -42.4%
completion p50     +73.6% 到 +76.3%
engine peak        -6.5% 到 -7.5%
```

第一次未优化二进制先导和高负载窗口都被判为无效，没有写入正式结果。Release Qwen uniform
恢复到 478.22–480.60 tokens/s 后才接受测量。

## 决策

保留 API 和实现，但它是 opt-in memory/median-TTFT policy，不成为吞吐默认。反例清楚说明：
减少 KV backing 不等于减少同等比例的整机峰值，也不等于更高吞吐。

下一步只增加一个实验轴：固定相同 8 请求，比较 1/2/4 桶，寻找 Pareto 点。原始证据在
[Experiment 114](../optimization-log/experiments/114-length-bucketed-cache.md)。
