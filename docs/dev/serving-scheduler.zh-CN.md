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
position和共享Storage地址都不变。它补齐了模型层的slot admission oracle；图解见
[单槽位prefill](slot-row-prefill.zh-CN.md)。

现在`ContinuousBatchScheduler`已经把这些动作串起来：固定`max_slots`行共享Cache，请求在step边界
入场，length/stop/cancel立即reset row，后来请求复用最低空slot。CPU/HIP、FP32/BF16、随机seed和
独立B1结果都通过。greedy HIP每step只复制一次`[slots]`选择结果。

这仍不是最终快路径。MI300X divergent workload只有串行reference的0.768×–0.878×；uniform
对照能达到1.511×–2.566×reference，却仍只有static batch的0.350×–0.768×。详细状态机、显存和
反例见[continuous slot scheduler](continuous-slot-scheduler.zh-CN.md)。

active-row compaction随后把空slot从模型输入中剔除，Release divergent五个shape提高
1.134×–1.348×，dummy rows降到0；图解见[active-row compaction](active-row-compaction.zh-CN.md)。
真实不同position的row仍逐rowB1，所以还不是最终并行实现。

positions-aware decode随后把`positions[A]`与`cache_rows[A]`送进RoPE、KV store和Attention，真实
divergent rows现在共享QKV/FFN/output batch。三组严格交替Release A/B提高1.295×–1.670×；图解见
[positions-aware decode](positions-aware-decode.zh-CN.md)。

同长度pending prompt现在也会调用`forward_prefill_cached_rows()`一起入场。uniform R8/S8的物理
prefill从8次降到1次，Release continuous达到static batch的87.4%；图解见
[batched slot prefill](batched-slot-prefill.zh-CN.md)。

真实模型也已经进入相同路径。官方runner覆盖Qwen/DeepSeek、短/长context、2/4 slot、补位、
KV利用率和峰值显存，并把Cache容量限制到当前请求真正需要的最大长度。Qwen 4/4逐token对齐；
DeepSeek仅1/4对齐，因此其余三组仍是明确失败。数据读法见
[官方连续推理矩阵](official-continuous-serving.zh-CN.md)。

公平的1/2/4/8 slot sweep随后固定同一批8条请求。它暴露并修复了“所有row位置归零、Storage仍
保留时误走首次prefill”的生命周期错误；同一48进程矩阵从30 pass/18 stable fail变成48/48
执行通过。长请求S8效率只有约40%，DeepSeek short仍有跨slot token分叉。见
[固定请求slot sweep](continuous-slot-sweep.zh-CN.md)。

为了定位而不污染普通计时，scheduler现在提供默认关闭的selection diagnostics。它记录logit来源、
真实batch、top-2和margin。DeepSeek反驳实验只关闭B2 prefill并保留B4/B8 decode，完整输出回到
S1，证明prefill shape是因果变量；默认B2仍因更接近PyTorch而保留。见
[低margin分叉诊断](continuous-divergence.zh-CN.md)。

显式`prompt_offsets`又把同一prompt放到B2 row0、row1并交换顺序；两行重复prompt的prefill
signature和完整输出逐值一致，排除了local row/stride/KV copy错误。见
[B2 prefill row审计](prefill-row-audit.zh-CN.md)。

## 8. 请求进入前的显式预热

使用exact BF16 GroupedQKV时，服务可以在开放admission前调用：

```cpp
auto report = model.prewarm_bf16_grouped_qkv(512);
```

调用前必须准备BF16 Attention权重、启用QKV Arena并注册当前环境的精确算法。报告把总时间、
grouped kernel初始化和每block device arguments准备分开。相同rows重复调用是no-op；移动模型或
重设Arena后需要重新预热。

这不是启动加速。Qwen/DeepSeek把约915/886ms移到admission前，首个已接纳请求比lazy grouped
快892/947ms，但两段相加仍约5.7秒。scheduler只有在明确管理“准备完成→开始接客”状态时才能
使用，普通默认路径不自动预热。

不要用HIPBLASLT_PRELOAD_KERNELS=1代替精确prewarm。Exp193在两个官方模型上把第一次
forward放慢约3.42–3.45倍，整个进程也慢约2.94–3.14倍。服务只能预热已经知道会使用的
rows、shape和plan；“全部加载以后总会用到”不是可接受的假设。

只注册第一个FFN精确solution也不够。Exp194的单算子快1.059×/1.032×，但cold与进程wall
全部更慢，steady也没有同时过线。服务启动合同因此只接受完整模型fresh-process证据，
不接受“第一个算子已经换成更快编号”作为ready证明。

Grouped gate/up是steady策略，不是启动策略。只有BF16 FFN Arena已开启、rows和backend环境
精确匹配且显式注册solution时才运行。服务可以在开放admission前用一次真实shape warm-up建立
每block plan；必须单独报告约57ms shared kernel setup，不能把它藏进TTFT口径。

同时启用Grouped QKV和gate/up时，先做QKV prewarm。实测这会把随后gate/up kernel setup降到
0.25ms以内；第一个完整dummy forward再建立gate/up的每block arguments。ready状态必须检查
两个registry和两个plan集合，不能只看其中一个已经warm。

批量prefill保存logits时，last模式会写B行，full模式会为每行选自己的最后token后写B行。
文件元素数必须是B×vocab。服务评测不能只检查B0，否则不同请求行的错误会被静默丢掉。

inference BTHD policy当前不用于写KV cache的prefill，也不用于value trace。scheduler若需要
cache admission，继续走旧路径；不能因为无cache的T512更快就删除fallback。

## 9. 测试位置

```text
tests/inference/scheduler_test.cpp
  CPU延迟到达、随机采样、取消幂等、Cache释放、错误与独立generate对齐

tests/ops/hip_ops_test.cpp
  HIP与CPU逐请求结果、取消行排除、Cache和调用指标对齐

tests/inference/hip_shape_matrix_test.cpp
  分叉row与单槽位prefill的CPU/HIP、零D2H、FP32/BF16和公共Storage对齐

benchmarks/end_to_end/benchmark_scheduler.cpp
  CPU/HIP 1/2/4/8请求的串行与静态batch基线

benchmarks/single_gpu/hf_continuous_matrix.py
  官方Qwen/DeepSeek多进程短/长context、slot、KV与显存矩阵

python/tests/test_hf_continuous_matrix.py
  suite轴、Cache公式、命令/schema和PyTorch比较边界合同
```
