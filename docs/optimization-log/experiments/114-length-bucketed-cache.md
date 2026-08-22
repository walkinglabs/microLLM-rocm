# Experiment 114：长度分桶省了一半 KV，但不能冒充吞吐优化

## 观察

Experiment 113 的长上下文 S8 只有 46.85% KV byte utilization。原因很直接：只有两条请求
需要 2064 token，但统一池给 8 个 slot 全部分配了 2064-token backing。

## 假设

把 8 个 slot 分成四个固定容量池：

```text
264×2, 520×2, 1040×2, 2064×2
```

请求进入能装下它的最小池。模型权重仍只有一份。假设是：KV backing 会显著下降；但拆成四个
B2 decode 可能损失原来的 B8 吞吐。

## 最小改动

新增 `LengthBucketedBatchScheduler`，它组合四个现有连续调度器。子调度器共享一个模型引用，
各自拥有固定容量 KV Cache。第一版没有 slot stealing、动态桶、多 Stream 或跨桶 batching。

CLI 显式记录桶配方和每条请求的路由。没有提供桶时，旧的统一池路径完全不变。

## 测量合同

- MI300X/gfx942，Release + hipBLASLt；
- Qwen2.5-0.5B 与 DeepSeek Distill Qwen 1.5B；
- 同一组 8 条长请求，同为 8 个总 slot；
- BF16 权重路径与 BF16 KV；
- 每个 policy 热身 1 次，每进程测 3 个完整 workload，3 个 fresh process；
- 保存每请求 token、TTFT/completion 原始数组、KV bytes、engine peak 与 GPU telemetry。

## 结果

| 模型 | Policy | KV MiB | KV 利用率 | tokens/s p50 | TTFT p50 ms | TTFT p95 ms | completion p50 ms | engine peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | uniform S8 | 193.50 | 46.85% | 479.03 | 64.43 | 64.45 | 162.83 | 2016.61 |
| Qwen | 4×B2 buckets | 91.13 | 99.49% | 275.75 | 27.68 | 77.06 | 287.05 | 1864.70 |
| DeepSeek | uniform S8 | 451.50 | 46.85% | 276.78 | 118.97 | 118.98 | 284.49 | 5441.08 |
| DeepSeek | 4×B2 buckets | 212.63 | 99.49% | 160.82 | 52.13 | 142.51 | 493.83 | 5088.17 |

两模型变化非常接近：

```text
KV backing             -52.91%
median TTFT             -56% 到 -57%
throughput              -41.9% 到 -42.4%
median completion       +73.6% 到 +76.3%
tail TTFT               约 +20%
engine peak             只下降 6.5% 到 7.5%
```

12/12 进程通过；uniform 与 bucketed 的两模型生成 token 都 exact。Qwen throughput 波动不超过
0.78%，DeepSeek 不超过 0.11%。

![Length bucket tradeoff](../assets/length-bucket-tradeoff.svg)

## 为什么 median TTFT 变好，completion 却变差

小桶的 prefill 很短，因此前四条短请求更早拿到第一个 token，拉低 median TTFT。但四个桶不能
一起做一个 B8 decode：

```text
uniform batch_decode_calls = 15
bucketed batch_decode_calls = 44
```

更多小 batch 和更多 launch 拉低总吞吐。最长桶最后才完成，所以 TTFT p95 和 completion 都变差。

## 为什么 engine peak 没有下降一半

KV backing 确实少了 52.91%，但 engine peak 还包含：

- 只保存一次、但体积很大的模型权重；
- logits 和层内临时 Tensor；
- allocator 已保留的可复用块。

所以“KV 少一半”不能改写成“整卡显存少一半”。正式记录只显示 6.5%–7.5% engine peak 下降。

## 决策

保留长度分桶作为显式、可选的 memory/median-TTFT policy，不改默认统一池。它适合 KV 预算紧张
且更在意 median 首 token 的场景，不适合吞吐或尾延迟优先的默认服务。

下一节点先测 1/2/4 桶粒度的 Pareto 曲线。两桶 B4 可能在 Cache 节省和 decode batch 效率之间
取得更好平衡；在测完之前，不直接跳到复杂的 paged Cache。

## 证据边界

这个 A/B 证明相同 microLLM 权重与请求在两种 policy 下 token exact。它不是 PyTorch 的可变位置
连续调度 oracle，也没有证明所有模型、请求分布或到达过程都保持 exact。
