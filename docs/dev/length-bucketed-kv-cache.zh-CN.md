# 按长度分桶：不要给每个请求都发一个最大号书包

## 先看见问题

连续推理需要保存每个请求已经读过的 Key 和 Value，这块显存叫 KV Cache。

原来的调度器只有一个 Cache 池。假设同时服务 8 个请求，其中两个请求可能长到 2064 token，
调度器就给 8 个 slot 都预留 2064 token：

```text
8 个 slot × 每个 2064 token = 16512 个 token-slot
```

这像班里只有两个同学要带大画板，却给所有同学都发了最大号书包。程序没有算错，但很多空间
一直空着。

## 新办法是什么

`LengthBucketedBatchScheduler` 准备几个不同大小的 Cache 池。请求需要的长度是：

```text
需要的 token 数 = prompt 长度 + 最多生成的 token 数
```

它把请求放进“能装下它的最小桶”。例如：

```text
桶 0：capacity 264，2 个 slot
桶 1：capacity 520，2 个 slot
桶 2：capacity 1040，2 个 slot
桶 3：capacity 2064，2 个 slot
```

8 个 slot 仍然一个不少，但总预留量变成：

```text
2×264 + 2×520 + 2×1040 + 2×2064 = 7768 个 token-slot
```

这里减少的是 KV Cache，不是模型权重。

## 模型为什么没有复制四份

结构如下：

```text
                         同一个 TransformerModel
                              ↑  ↑  ↑  ↑
LengthBucketedBatchScheduler ─┼──┼──┼──┼─ 只保存引用
                              │  │  │  │
                         四个固定容量调度器
                              │  │  │  │
                         四块独立 KV Cache
```

每个子调度器都引用同一个 `TransformerModel`。权重由模型拥有；子调度器只拥有自己的 slot、
logits 暂存和 KV Cache。因此 `resident_weight_bytes` 只能计算一次。

## 一条请求怎样走

以 `prompt=512`、`max_new_tokens=8` 为例：

1. 外层算出 `512 + 8 = 520`。
2. capacity 264 装不下。
3. capacity 520 刚好装下，因此选择桶 1。
4. 子调度器给请求分配一个本地 ID 和 slot。
5. 外层再分配一个全局 ID。用户永远只看见全局 ID。
6. `request(global_id)` 先查路由表，再读取正确的子调度器。

全局 ID 很重要。每个子调度器的第一个本地 ID 都是 1；如果直接暴露本地 ID，四条不同请求
会看起来都叫 1。

## 为什么第一版不“借用”大桶的空 slot

如果小桶已经排队，而大桶还有空位，第一版仍让短请求等待小桶。这叫“没有 work stealing”。

这不是最终最聪明的策略，但它有三个优点：

- 路由规则固定，同一输入总会进入同一桶；
- 显存预算容易手算和测试；
- 我们可以先单独测量节省多少显存、损失多少吞吐，再决定是否增加借用规则。

代价也必须写清：请求长度分布与 slot 配方不匹配时，一个桶可能排队，另一个桶却空着。

## 为什么输出可能和单池不同

分桶没有改变公式，却改变了同一次矩阵乘法里有多少行：

```text
单池：不同长度的 8 行可能一起 decode
分桶：每个长度桶分别计算 2 行
```

BF16 的并行归约顺序和 hipBLASLt 方案可能随 shape 改变。浮点加法又不是严格满足结合律，所以
logits 最后几位可能改变；如果第一、第二名非常接近，greedy token 也可能改变。

因此验收分成三层，不能只说“能运行”：

1. 状态机：提交、取消、完成、slot 释放和 ID 映射正确；
2. 数值：逐请求 token 对齐，并保留不一致的位置；
3. 性能：分别报告 KV bytes、引擎峰值显存、tokens/s、TTFT 和完成延迟。

## 怎样使用

统一 Cache 的旧接口不变。按长度分桶是显式选择：

```bash
build/apps/microllm_hf_infer \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokens 9707,1879 \
  --device hip \
  --new-tokens 0 \
  --warmup 1 \
  --steps 3 \
  --bf16-ffn true \
  --bf16-attention true \
  --workload continuous \
  --kv-cache-dtype bf16 \
  --continuous-slots 8 \
  --continuous-prompt-lengths 256,256,512,512,1024,1024,2048,2048 \
  --continuous-new-token-lengths 8,8,8,8,16,16,16,16 \
  --continuous-cache-buckets 264:2,520:2,1040:2,2064:2
```

`capacity:slots` 从小到大书写，各桶 slot 数之和必须等于 `--continuous-slots`。JSON 会记录：

- `bucketed_cache`：是否启用分桶；
- `continuous_cache_buckets`：完整桶配方；
- `request_bucket_indices`：每条请求实际进入哪个桶；
- `allocated_cache_bytes`：所有桶的 KV backing storage 之和；
- `peak_active_cache_bytes`：调度过程中采样到的有效前缀峰值；
- 吞吐和每请求延迟数组。

## 测试在哪里

- `tests/inference/scheduler_test.cpp`：最小桶路由、全局 ID、取消、排队、显存公式和独立生成对齐；
- `python/tests/test_hf_continuous_matrix.py`：CLI 字符串、8-slot 不变量和理论 Cache 下降；
- `benchmarks/single_gpu/hf_continuous_matrix.py --suite length-buckets`：官方 Qwen/DeepSeek A/B 矩阵。

CPU sanitizer 已覆盖组合调度器，MI300X Release 全套测试为 318/318。正式 A/B 使用 12 个
fresh process，并记录了 GPU 空闲窗口。

结果不是“分桶全面更快”：两模型 KV backing 都减少 52.91%，median TTFT 改善 56%–57%，但
吞吐下降约 42%，completion p50 增加 74%–76%。所以它保留为显式可选策略，默认仍是统一池。
原始数组、图和解释见 [Experiment 114](../optimization-log/experiments/114-length-bucketed-cache.md)。

后续 1/2/4 桶扫描发现两个 B4 桶是当前固定请求的折中点：KV 少 37.4%，median TTFT 改善约
35%，吞吐损失约 14%。但尾延迟仍变差，且请求分布刚好均匀，所以不能自动设成默认。详见
[Experiment 115](../optimization-log/experiments/115-bucket-pareto.md)。

## 当前边界

- 各桶按顺序提交 GPU 工作，还没有多 stream 并发；
- 不支持跨桶 batching；
- 不支持动态增加、删除桶；
- 不支持空闲大桶 slot 借给小桶；
- 分桶数值是否与单池 token 完全一致，必须由每个模型的真机记录回答。

这些限制是下一轮实验的起点，不会被包装成已经解决。

## 可选的兼容大桶溢出

Experiment 116 证明固定桶会让短请求在小桶外排队，即使大桶还有空位。框架现在提供显式候选：

```text
--continuous-bucket-overflow true
```

规则只在提交时执行：最小兼容桶的“活跃+等待”请求已达到 slot 数时，寻找第一个仍有即时容量的
更大兼容桶。请求提交后不会迁移；大桶也满时仍回到最小桶排队。默认值是 `false`，正式
MI300X P95 数据通过前不会自动启用。

长请求无法装进小桶，因此该候选只可能修复 short-heavy，不会解决 long-heavy。这个反例是 API
合同的一部分，不是待删除的难看数据。

正式 54 进程证明：short-heavy 相对固定桶吞吐提高约 13%，TTFT P95 下降 61%–62%，completion
P95 下降约 40%；不发生溢出的 long/delayed 与固定路径基本一致。但候选仍未追平 uniform，
因此默认继续关闭。详见
[Experiment 117](../optimization-log/experiments/117-compatible-overflow.md)。
