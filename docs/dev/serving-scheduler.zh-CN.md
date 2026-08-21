# Serving Scheduler：让多位同学轮流使用同一个模型

## 1. 为什么需要调度器

以前的`generate()`一次只处理一条请求。真实服务里，请求会在不同时间到达：有人只生成
2个token，有人要生成20个；prompt长度也不同。

可以把模型想成一位老师，KV Cache是每位同学自己的草稿本。调度器要保证：

- 新同学可以中途加入；
- 每个人只读写自己的草稿本；
- 先完成的人及时释放草稿本；
- 调度后的答案与每个人单独调用`generate()`相同。

## 2. 四个状态

```text
PendingPrefill → Decoding → Completed
       └───────────────→ Cancelled
```

- `PendingPrefill`：prompt还没有写入Cache；
- `Decoding`：已经有Cache，每个scheduler step生成一个token；
- `Completed`：达到`max_new_tokens`，Cache立即释放。
- `Cancelled`：调用者不再需要答案；保留已经生成的token，同时立即释放Cache。

`Completed`和`Cancelled`都是终态。再次取消它们会返回`false`，不会重复计数，也不会改变
结果。未知请求ID会抛出错误，避免把写错ID误认为“取消成功”。

状态只说明“还在不在运行”，`completion_reason`再说明为什么结束：达到长度、生成stop token，
还是被调用者取消。可以把stop token理解成模型写出的“句号”：这个token属于答案，但写完后
不应再让模型继续写下一页。

`ReferenceScheduler`每一步按请求顺序逐个调用模型。它故意不做跨请求batch，因为后续
优化必须先有一个最容易检查的正确答案。

## 3. C++例子

```cpp
microllm::inference::ReferenceScheduler scheduler(model);

auto alice = scheduler.submit(
    {1, 2, 3}, {.max_new_tokens = 4,
                .temperature = 0.0F,
                .top_k = 1,
                .kv_cache_layer_dtypes = {},
                .stop_tokens = {2}});

scheduler.step();  // Alice生成第一个token
bool alice_was_cancelled = scheduler.cancel(alice);

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
- 提交、完成和取消请求数。
- 其中有多少请求由stop token提前完成。

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

## 7. Admission bucketing

`AdmissionBatchScheduler`把当前等待请求按“prompt长度、生成配置、seed、Cache策略”分组。
兼容请求走`generate_batch()`，不兼容请求走B1；下一次`drain()`可以接收后来到的请求。

它解决的是“入场时怎样分组”，没有解决“生成过程中怎样腾出并补充slot”。B4兼容组在HIP
约1260 token/s，但8/16请求拆成2/4个B4组后吞吐保持平台。

现在static batch允许某一row先生成stop token。它会停止追加该row的答案，但仍用dummy token
维持共同position，直到整组结束。这保证答案正确，却没有释放物理slot。真正的slot refill还要
支持清空某一row的旧K/V、写入新prompt和独立position。

`KVCache::clear_row(row)`已经能在CPU/HIP清空某一slot的完整capacity，其他row不变且没有
GPU payload copy。它故意不修改共同`position()`：清掉旧草稿本不等于新同学已经写到相同页数。

现在Cache已经保存`row_positions[B]`。如果全部位置相同，旧`position()`继续工作；一旦分叉，
旧接口明确报错，调用者必须逐row读取。`reset_row()`同时清Storage并把该row位置归0。模型
提供`forward_cached_rows()`作为正确性优先的计算路径：每个row建立共享原Storage的B1 view，
使用自己的RoPE位置、K/V写入位置和可见prefix，再合并logits。uniform位置仍走原batch快路径。

这条路径会串行执行B个B1 forward，并用D2D合并输出，因此它是positions-aware HIP Kernel和
continuous refill scheduler的oracle，不是最终性能实现。详细图解见
[不同页数的KV row](divergent-kv-rows.zh-CN.md)。

空row现在也能调用`forward_prefill_cached_row()`接收一个新`[1,T]` prompt。第一版先在临时B1
Cache里跑已验证的full prefill，再把有效K/V逐层、逐head D2D复制进目标row。原有row的K/V、
position和共享Storage地址都不变。它补齐了模型层的slot admission oracle，但scheduler还没有把
结束、清空、补位和decode自动串起来。图解见[单槽位prefill](slot-row-prefill.zh-CN.md)。

## 8. 测试位置

```text
tests/inference/scheduler_test.cpp
  CPU延迟到达、随机采样、取消幂等、Cache释放、错误与独立generate对齐

tests/ops/hip_ops_test.cpp
  HIP与CPU逐请求结果、取消行排除、Cache和调用指标对齐

tests/inference/hip_shape_matrix_test.cpp
  分叉row与单槽位prefill的CPU/HIP、零D2H、FP32/BF16和公共Storage对齐

benchmarks/end_to_end/benchmark_scheduler.cpp
  CPU/HIP 1/2/4/8请求的串行与静态batch基线
```
