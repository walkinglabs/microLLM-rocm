# Continuous slot scheduler：有人离开后，马上让下一位入座

## 1. 它解决什么问题

静态batch像老师点名后锁门：4位同学一起开始，必须等这一组全部结束，下一位才能进来。真实服务
中，请求长短不同。短请求先结束时，空着GPU位置很浪费。

Continuous batching把batch看成固定数量的座位：

```text
step 1: slot 0 = A，slot 1 = B
step 2: A完成，slot 0清空；B继续
step 3: slot 0 = 新请求C；B继续
```

现在公开类是：

```cpp
ContinuousBatchScheduler scheduler(
    model, {.max_slots = 2, .kv_cache_dtype = DType::BFloat16});
```

## 2. 一步里面发生什么

每次`step()`严格按下面顺序：

1. 按提交顺序把pending请求放进最低编号的空slot；
2. 新请求调用单row prefill，只写自己的prompt；
3. 从每个已占用slot的logits选择一个token；
4. 达到长度、stop token或取消的请求清空自己的整行Cache；
5. 仍在生成的请求用`forward_cached_rows()`继续一步；
6. 没有真实请求的row用dummy token维持固定batch，计算后立刻reset。

新请求最早在下一次scheduler step补位。这个边界是故意固定的，避免同一步内反复完成、补位造成
难以解释的循环。

## 3. 为什么使用一块共享Cache

共享Cache的shape是：

```text
[max_slots, kv_heads, model_capacity, head_dimension]
```

每个row有独立position。请求完成时只清自己的row，底层Storage地址和总容量不变。因此补位不需要
重新申请整块Cache。

指标分两本账：

- `allocated_cache_bytes`：固定座位一共预留多少；
- `active_cache_bytes`：当前各请求实际写了多少token。

终止后active可以回到0，但allocated仍保留，供下一位请求复用。这不是泄漏。

## 4. 为什么第一版可能更慢

第一版的目标是把状态机做对。只要各row position不同，`forward_cached_rows()`就逐row运行B1模型；
空slot还会产生dummy row。MI300X Release tiny benchmark中continuous/reference只有0.748×–0.858×。

这不是“continuous batching没有用”，而是证明当前计算层仍缺：

```text
positions[B]
  → 一次并行RoPE
  → 一次并行K/V store
  → 每row只看自己的prefix
  → 一次并行Attention
```

当前scheduler是以后并行Kernel的正确答案，不能拿它宣称吞吐提升。

如果所有请求等长、同时入场并保持相同position，Release uniform对照能达到串行reference的
1.43×–2.36×。这证明batch计算本身有用。但它仍只有static batch的30.8%–68.0%，因为每个prompt还逐row prefill，
每步也有调度和选择开销。

## 5. 怎样检查它没有串请求

测试同时运行独立B1 oracle，并检查：

- A、B、后来到的C各自token完全相同；
- A结束后C复用slot 0；
- B的Cache和生成不中断；
- FP32/BF16 Cache都通过；
- 随机采样使用每请求自己的seed；
- stop、cancel和长度完成都立即释放row；
- KV策略不匹配明确报错；
- HIP greedy选择每step只有一次`[slots]` D2H；
- slot利用率、dummy row、refill和active/allocated Cache都有计数。

代码测试位置：

```text
tests/inference/scheduler_test.cpp
tests/ops/hip_ops_test.cpp
benchmarks/end_to_end/benchmark_scheduler.cpp
```

实验记录见 [Experiment 096](../optimization-log/experiments/096-continuous-slot-scheduler.md)。
