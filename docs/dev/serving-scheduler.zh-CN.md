# Serving Scheduler：让多位同学轮流使用同一个模型

## 1. 为什么需要调度器

以前的`generate()`一次只处理一条请求。真实服务里，请求会在不同时间到达：有人只生成
2个token，有人要生成20个；prompt长度也不同。

可以把模型想成一位老师，KV Cache是每位同学自己的草稿本。调度器要保证：

- 新同学可以中途加入；
- 每个人只读写自己的草稿本；
- 先完成的人及时释放草稿本；
- 调度后的答案与每个人单独调用`generate()`相同。

## 2. 三个状态

```text
PendingPrefill → Decoding → Completed
```

- `PendingPrefill`：prompt还没有写入Cache；
- `Decoding`：已经有Cache，每个scheduler step生成一个token；
- `Completed`：达到`max_new_tokens`，Cache立即释放。

`ReferenceScheduler`每一步按请求顺序逐个调用模型。它故意不做跨请求batch，因为后续
优化必须先有一个最容易检查的正确答案。

## 3. C++例子

```cpp
microllm::inference::ReferenceScheduler scheduler(model);

auto alice = scheduler.submit(
    {1, 2, 3}, {.max_new_tokens = 4,
                .temperature = 0.0F,
                .top_k = 1,
                .kv_cache_layer_dtypes = {}});

scheduler.step();  // Alice生成第一个token

auto bob = scheduler.submit(
    {7, 8}, {.max_new_tokens = 2,
             .temperature = 0.0F,
             .top_k = 1,
             .kv_cache_layer_dtypes = {}});

scheduler.run_until_idle();
auto alice_result = scheduler.request(alice);
auto bob_result = scheduler.request(bob);
```

每个请求可以使用自己的随机种子和KV dtype策略。

## 4. 指标怎样读

`SchedulerMetrics`报告：

- scheduler执行了多少step；
- prefill/decode模型调用数；
- 同时活跃请求峰值；
- 当前和峰值Cache字节；
- 提交和完成请求数。

Cache字节是引擎Storage实际分配，不是只看active view得到的估算。

## 5. 当前不能做什么

Reference scheduler中，每条请求仍单独运行B=1 forward：

```text
request A forward
→ request B forward
→ request C forward
```

所以请求增加不会带来GPU batch吞吐扩展。基准中HIP 1/2/4/8请求都约331 token/s，正好
证明它是串行reference。

下一版slot scheduler才会尝试：

```text
多个可兼容请求
→ 组成一次batched forward
→ 按slot拆回答案
```

它必须逐请求对齐本页reference，包括延迟到达、随机状态、完成顺序和Cache释放。

## 6. 已有的静态batch积木

`generate_batch()`已经能把等长、同配置请求放进一次`[B,T]`/`[B,1]` forward。prompt内容
可以不同，CPU/HIP逐行与独立生成对齐。MI300X tiny B8达到7.31×串行reference、90.7%
扩展效率。

它仍不能接收晚到请求或为提前结束的请求补新slot，所以只是continuous batching的计算积木。

## 7. 测试位置

```text
tests/inference/scheduler_test.cpp
  CPU延迟到达、随机采样、状态、错误与独立generate对齐

tests/ops/hip_ops_test.cpp
  HIP与CPU逐请求结果、Cache和调用指标对齐

benchmarks/end_to_end/benchmark_scheduler.cpp
  CPU/HIP 1/2/4/8请求的串行与静态batch基线
```
