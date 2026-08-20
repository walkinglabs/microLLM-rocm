# 从 0.1917× 到 1.0×：在 MI300X 上从零优化一个 C++/HIP LLM 引擎

> 这是一篇会随着实验继续生长的博客。绿色不是“计划成功”，只有实际测量并通过
> 正确性门的改动才会出现在绿色 running-best 上。

![当前优化进度](assets/progress.svg)

## 0. 我们到底在优化什么

microLLM-rocm 已经可以：

- 用纯 C++/HIP 构造 Tensor、算子和自动求导图；
- 在 MI300X 上加载官方 Qwen2.5-0.5B；
- 运行 DeepSeek-R1-Distill-Qwen-1.5B；
- 让 logits、生成 token、loss 和参数更新与 PyTorch 对齐；
- 使用两张 GPU 做 RCCL 数据并行正确性实验。

但“输出正确”与“像成熟框架一样快”是两个完全不同的终点。

在固定 FP32、多步、排除热身的 MI300X 对比中：

| Workload | microLLM | Python/PyTorch | microLLM/PyTorch |
|---|---:|---:|---:|
| Qwen train | 7.30 token/s | 51.32 token/s | 0.1422 |
| Qwen generate | 18.77 token/s | 70.18 token/s | 0.2675 |
| DeepSeek Distill train | 5.79 token/s | 26.23 token/s | 0.2209 |
| DeepSeek Distill generate | 10.02 token/s | 62.40 token/s | 0.1606 |

四项几何平均只有：

```text
0.191660× PyTorch
```

经过十四个保留实验，当前重复进程 score 已到 `2.478439×`。这不是估计值：它来自同一张
MI300X、同一组模型权重、同样的 2 次热身和 5 次正式测量。

因此，本专项的故事不是“我们已经超过 PyTorch”，而是：

> 一个正确但很慢的教学引擎，怎样经过可反驳的系统实验，一步步长成一个在固定
> AMD GPU 矩阵上接近成熟框架的引擎？

这里的最终 `1.0×` 只表示固定 GPU、模型、dtype、batch、context 和 token 的 selected
matrix parity。它不代表所有模型、所有 shape 或所有 AMD GPU。

## 1. 从 autoresearch 学到的不是“让 Agent 随便改代码”

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的核心不是一句
“AI 自动研究”，而是一个非常克制的实验系统：

```text
固定时间预算
固定评价指标
一次做一个实验
成功保留
失败回退
结果持续写入表格
用图展示 running best
```

它的 `progress.png` 里，灰点是被丢弃的实验，绿点是保留的实验，绿色阶梯线是
running best。真正有价值的不是曲线看起来不断改善，而是失败点没有被删除。

microLLM 的问题更复杂。模型质量实验可以使用一个 `val_bpb`，系统实验至少需要：

- 四项吞吐；
- 四项峰值显存；
- logits、loss、gradient 和 generated token；
- Kernel、copy、allocation 和 launch 时间线；
- reference、readable 和 optimized 三条路径。

所以我们借用实验循环，不照搬指标。

## 2. 为什么要先固定测量，而不是马上写 Kernel

最早的一次官方模型对比只测了第一个训练 step，结果看起来 microLLM 比 PyTorch
快约 3 倍。如果停在这里，我们可能会写出一篇错误的“自研框架击败 PyTorch”。

加入两次 warm-up、再测五步后，结果变成：

```text
Qwen train       microLLM 7.30   PyTorch 51.32
DeepSeek train   microLLM 5.79   PyTorch 26.23
```

原来的“领先”只是 PyTorch 首次 Kernel 初始化成本。

因此本项目固定：

```text
load
→ 2 complete warm-up iterations
→ reset peak allocator counter
→ 5 measured iterations
→ only measured interval enters throughput
```

setup、warm-up 和 measured 必须分别报告。不能把不喜欢的时间藏起来，也不能把
一次首次启动写成 steady-state。

## 3. 先画出当前系统，而不是凭感觉优化

![当前瓶颈图](assets/bottleneck-map.svg)

优化起点的训练路径：

```text
token
→ Embedding
→ [RMSNorm → QKV → RoPE → QK → Softmax → PV → O]
→ [RMSNorm → gate/up → SwiGLU → down]
→ output head
→ CrossEntropy
→ backward
→ AdamW
```

优化起点的生成路径：

```text
token
→ 每层计算 Q/K/V
→ K/V 搬回 CPU
→ CPU 拼接旧 cache
→ 搬回 GPU
→ GQA 再搬回 CPU 展开
→ 再搬回 GPU
→ Attention
→ tied output weight 整体转置复制
→ 完整 logits 搬回 CPU 做 argmax
```

程序能输出正确 token，但这条数据路线显然不适合高性能 GPU。

## 4. Baseline profiler 告诉了我们什么

### 4.1 Qwen 推理

代表性 rocprof：一次完整 forward、一次 warm-up generate、一次 measured generate。

| Kernel | Calls | Total | Share |
|---|---:|---:|---:|
| strided transpose copy | 323 | 119.25 ms | 43.43% |
| RMSNorm | 539 | 103.45 ms | 37.67% |
| largest hipBLASLt shape | 792 | 17.89 ms | 6.52% |
| readable batched matmul | 538 | 6.58 ms | 2.40% |

另外还有：

```text
hipMalloc 7404
hipFree   7403
hipMemcpy 2712
ordinary Kernel launch 4099
```

结论非常反直觉：GEMM 不是第一热点。继续只调 GEMM，端到端不会发生根本变化。

### 4.2 Qwen 训练

| Kernel | Share |
|---|---:|
| CrossEntropy backward | 49.52% |
| CrossEntropy forward | 26.21% |
| strided transpose copy | 8.43% |
| RMSNorm backward | 7.48% |
| RMSNorm forward | 2.76% |
| AdamW | 1.49% |

这说明训练第一刀应该砍 CrossEntropy，而不是 AdamW。

## 5. 第一个结构性错误：GPU 上的串行 reduction

当前 CrossEntropy forward/backward 由一个 GPU 线程循环完整词表。Qwen 词表是
151,936，三个 token 就需要一个线程反复执行几十万次 max、exp 和 sum。

RMSNorm 也类似：一个线程处理整行 hidden。Qwen width 896，DeepSeek width 1536。

正确的 GPU 思路应是：

```text
一行交给一个 block
→ 每个线程处理多个元素
→ wave reduction
→ block reduction
→ 必要时第二阶段 reduction
```

Step 01 和 Step 03 分别处理这两个问题。它们必须先保留简单 CPU 参考，再新增
readable parallel HIP 和 optimized candidate。

## 6. 第二个结构性错误：把 view 变成 544 MB 的复制

Qwen 使用 tied embedding。输出 logits 时，Embedding weight 既当输入表，又当输出
投影矩阵。

优化前的实现：

```text
[151936, 896]
→ transpose view [896, 151936]
→ contiguous copy
→ GEMM
```

每个 token 产生约 544 MB 转置副本。训练 backward 中每个 Linear 权重也会执行
类似转置复制。

真正的解决办法不是缓存一个永久副本——那会让 tied weight 更新一致性变复杂——
而是让 GEMM 接受 transpose/stride 描述。hipBLASLt 可以把 transpose 当作 layout，
不移动数据。

Step 02 的验收不是“Kernel 变快”，而是 profiler 中整个权重 strided copy 消失，
同时 logits 和梯度保持一致。

这个问题可以把 GEMM 想成读一张表。过去为了“横着读”，程序先把整张大表抄成
另一张表；现在我们只告诉 GEMM：“请交换行号和列号来读原表”。数字没有搬家，
只是读取公式变了。

## 7. 第三个结构性错误：KV Cache 不是 Cache

真正的 Cache 应当：

```text
初始化时分配一次
每步原地写一个 position
过去的数据地址不变
```

当前 Cache 每步重新创建更大的 Tensor，并通过 CPU 拼接。GQA 又把少量 KV head
复制成所有 query head。

目标结构：

```text
K/V [B, KVH, max_T, head_dim]
length
current K/V → device slice write
query_head → 直接映射 kv_head
```

完成后，生成循环里不应有 Tensor payload H2D/D2H。只有最终需要显示文字时，才
把 token IDs 取回 CPU。

## 8. 为什么 allocator 本身也会拖慢模型

当前每个临时 Tensor 都直接 hipMalloc/hipFree。短 Qwen 推理就出现七千多次申请和
释放。

`hipFree` 还可能等待尚未完成的 GPU 工作。于是一个普通 C++ 析构函数可能变成
隐藏同步点。

Step 06 将按顺序引入：

```text
size-class cache
→ per-device pool
→ stream-aware retirement
→ workspace arena
→ stable-address graph memory plan
```

allocator 改动风险很高：复用过早会让另一个 Stream 读取已经被覆盖的数据。因此
正确性测试必须包含 Event 未完成、异常退出、多 Stream 和峰值统计。

## 9. 为什么低精度不放在最前面

FP8 不会自动修复：

- 单线程 CrossEntropy；
- 单线程 RMSNorm；
- CPU KV 拼接；
- tied weight 整体复制；
- 数千次 malloc/free。

更糟的是，当前 FP8 Linear 每次 forward 还会重新量化 weight。

因此低精度是独立 track：

```text
先完成 FP32 数据路线
→ BF16 官方整网
→ FP32 master + BF16 compute
→ cached FP8 weights
→ dynamic activation scale
→ 同 dtype PyTorch 对比
```

FP32 running-best 和 BF16/FP8 running-best 不能连成一条线。

## 10. 最终希望长成什么架构

```text
Model weights
  ├─ logical layout + transpose flags
  ├─ cached hipBLASLt plan/workspace
  └─ optional BF16/FP8 packed copy

Training
  ├─ parallel/fused CE
  ├─ parallel RMSNorm
  ├─ transpose-aware backward GEMM
  ├─ reusable grad/activation arena
  └─ device-native optimizer

Inference
  ├─ preallocated K/V
  ├─ direct GQA indexing
  ├─ fused prefill/decode Attention
  ├─ device sampling
  └─ stable-address HIP Graph
```

可选后端可以使用 hipBLASLt、Composable Kernel 或其他 ROCm 原语，但核心依然保留
CPU reference 和仓库自己的 readable HIP 路径。

## 11. Experiment 001：第一条绿色阶梯

Step 01 把 CrossEntropy 从单线程循环改成 block-parallel reduction。结果：

```text
CE Kernel share          75.73% → approximately 0.62%
Qwen train               7.30 → 24.03 token/s
DeepSeek train           5.79 → 13.30 token/s
four-workload score      0.191660 → 0.318328
```

这是一次 `keep`。它没有改变推理路径，因此两个 generation row 保持在正常波动内。

更重要的是，新 trace 把下一层瓶颈暴露出来：

```text
strided transpose copies   33.55%
RMSNorm backward           30.54%
RMSNorm forward            11.14%
```

因此下一次实验不是继续微调 CE，而是 Step 02：让 tied output 和 backward GEMM 使用
逻辑 transpose，删除完整权重复制。

## 12. Experiment 002：不抄整张表，只改变读法

这次我们给矩阵乘法增加两个布尔条件：左矩阵是否转置、右矩阵是否转置。
`false/true` 的意思不是先创建一个转置 Tensor，而是在取第 `k` 个数时直接计算正确
下标。CPU reference、可读 HIP 和 hipBLASLt 都遵守同一个公式：

```text
C = op(A) × op(B)
op(A/B) 可以是原样，也可以是逻辑转置
```

为什么测试四种组合？因为只测 tied head 的 `A × Bᵀ`，很容易把 hipBLASLt 的
行主序、列主序碰巧写对。NN、NT、TN、TT 四种都通过，才能说明接口本身成立。

自动求导也不能偷偷抄表。以 tied head `Y = H × Wᵀ` 为例：

```text
dH = dY × W
dW = dYᵀ × H
```

新的 backward 直接把这两个式子交给 transpose-aware GEMM。图测试还明确检查 forward
图中没有 `transpose` 和 `contiguous` 节点。PyTorch oracle 同时比较输出、`dH` 和
`dW`，不是只比较最后一个 loss。

固定测量结果：

| Workload | Step 01 | Step 02 | 本步加速 | 当前 PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 24.03 | 38.77 token/s | 1.61× | 0.7554 |
| Qwen generate | 18.85 | 35.35 token/s | 1.88× | 0.5038 |
| DeepSeek train | 13.30 | 22.36 token/s | 1.68× | 0.8524 |
| DeepSeek generate | 10.05 | 10.15 token/s | 1.01× | 0.1626 |

```text
four-workload score       0.318328 → 0.479227
strided-copy Kernel time  62.33 ms → 2.16 ms
strided-copy calls         1302 → 624
```

DeepSeek 生成几乎没变不是坏消息。它说明收益来自真正删掉的 tied-weight 转置路径，
而不是测量噪声让所有数字一起“变漂亮”。Qwen 推理峰值从约 2.52 GB 降到 1.98 GB，
也符合“没有再创建一块大副本”的解释。

新 trace 中 RMSNorm forward/backward 已占 Kernel 时间 `64.31%`。因此 Step 03 将只改
RMSNorm，而不会顺便加入 allocator、KV Cache 或低精度。

## 13. Experiment 003：让一整组线程合作算平均值

RMSNorm 要先算一行数字的平方平均值。旧 Kernel 像让一个学生独自把整排 1536 个数
相加；GPU 虽然有许多线程，其他线程却没有参与。

新 forward 把一行交给一个 256-thread block：

```text
每个线程读取几列
→ 每个线程得到一个局部平方和
→ block reduction 合成整行平方和
→ 所有线程并行写归一化结果
```

backward 还需要 weight gradient。如果让每行都抢着原子加同一个 weight 位置，新的
瓶颈可能只是从串行循环变成原子竞争。因此我们先保存每行一个 `inverse_rms`，再启动
第二个 Kernel：一个线程负责一个 weight 列，并按行累加。这样写入位置互不争抢。

测试不是只用 3 个数。专门的 HIP gate 覆盖：

```text
rows   = 1, 3, 32
width  = 16, 384, 512, 896, 1536
data   = 0、正数、负数、较大值
check  = forward、input gradient、weight gradient、zero host transfer
```

固定端到端结果：

| Workload | Step 02 | Step 03 | 本步加速 | 当前 PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 38.77 | 71.06 token/s | 1.83× | 1.3845 |
| Qwen generate | 35.35 | 57.32 token/s | 1.62× | 0.8168 |
| DeepSeek train | 22.36 | 47.91 token/s | 2.14× | 1.8269 |
| DeepSeek generate | 10.15 | 18.60 token/s | 1.83× | 0.2980 |

```text
four-workload score      0.479227 → 0.885816
RMSNorm Kernel time      75.85 ms → 1.55 ms
RMSNorm Kernel share      64.31% → 3.59%
all Kernel time         117.94 ms → 43.25 ms
```

Qwen 和 DeepSeek 的训练 ratio 已超过 `1.0`，但这里必须克制：输入只有 4 个 token，
batch 是 1，dtype 是 FP32。这只能说明固定短序列测试领先，不能说明长上下文完整训练
已经超过 PyTorch。两个 generation ratio 仍低于 1，也说明系统工作远未结束。

至此 M1 的三个串行/复制热点都完成。下一阶段转向数据路线：KV Cache、GQA 展开、
sampling 和 allocator。

## 14. Experiment 004：真正的 Cache 不能每次重抄旧作业

旧 KV Cache 名字里虽然有 Cache，行为却更像每写一个新字，就把整页旧内容重抄到
一张更大的纸上。GQA 还会把少量 KV head 复制成和 query head 一样多。

新的数据结构只分配一次：

```text
backing storage [1, kv_heads, request_capacity, head_dim]
logical prefix [1, kv_heads, current_length, head_dim]
```

每来一个 token，HIP Kernel 只写 `position` 那一小行。Tensor 的 shape 显示当前有效
长度，底层 Storage 地址不变。计算 Attention 时，query head 通过整数除法找到自己的
KV head，不再创建展开后的 K/V Tensor。

这里出现了一个很有价值的失败。第一版按模型理论最大 context 分配：

```text
Qwen peak       1.98 GB → 2.78 GB
DeepSeek peak   7.11 GB → 14.63 GB
```

速度变快不代表设计可以接受。生成请求一开始就知道最多会用多少 token，所以保留版
按 `prompt + max_new_tokens` 分配。修正后 Qwen/DeepSeek peak 回到约 1.98/7.11 GB，
Storage 地址仍然稳定。

固定结果：

| Workload | Step 03 | Step 04 | 本步加速 | 当前 PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 71.06 | 72.33 token/s | 1.02× | 1.4092 |
| Qwen generate | 57.32 | 85.64 token/s | 1.49× | 1.2203 |
| DeepSeek train | 47.91 | 49.47 token/s | 1.03× | 1.8864 |
| DeepSeek generate | 18.60 | 35.79 token/s | 1.92× | 0.5735 |

```text
four-workload score  0.885816 → 1.167931
hipMemcpy calls           2712 → 600
copyBuffer calls          2269 → 253
```

进度线第一次越过 1.0，但右边四根柱子更重要：DeepSeek generation 仍只有 PyTorch 的
57.4%。因此“综合 parity”不能被写成“所有任务都超过 PyTorch”。

长一点是否还能运行？Qwen 的 1/32/128/512 新 token 实测分别约为
`53.2/97.5/90.7/68.4 token/s`，峰值显存随请求容量缓慢增加。它没有证明完整 32K
context，但排除了“只在一步 decode 生效”的解释。

## 15. Experiment 005：为了找一个最大值，不要搬回十五万个数

greedy 生成只需要回答：“哪个 token 的 logit 最大？”旧代码却把完整 vocabulary
搬回 CPU。Qwen 每步约 151,936 个 float，最后只留下一个 int32。

新 argmax 让 256 个线程分段寻找最大 `(value, index)`，再做 block reduction。相同
最大值永远选择较小 index，因此结果与 CPU `max_element` 一致。遇到 NaN/Inf 时设备
写 `-1`，C++ 读到后仍走原来的错误路径。

输出不是普通 CPU 整数，而是 GPU 上的 `[1,1] int32 Tensor`：

```text
GPU logits
→ GPU argmax
→ 4-byte token ID copy for returned C++ vector
↘ same GPU token Tensor goes directly into next Embedding
```

固定结果：

| Workload | Step 04 | Step 05 | 本步加速 | 当前 PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen generate | 85.64 | 93.34 token/s | 1.09× | 1.3300 |
| DeepSeek generate | 35.79 | 38.99 token/s | 1.09× | 0.6249 |

```text
four-workload score       1.167931 → 1.219170
generated-loop D2H records       9 → 1
```

为什么不顺便写 device top-k RNG？greedy 和随机采样是两种不同合同。这个实验只对
固定 benchmark 中实际使用的 greedy 路径负责；固定 seed 的随机 top-k 测试继续跑
CPU reference。把它写成“device sampling 全部完成”会夸大事实。

## 16. Experiment 006：内存池首先是生命周期问题

一个简单内存池像把用完的作业纸放回抽屉，下次要同样大小就直接拿。但 GPU 有个
麻烦：CPU 觉得 Tensor 析构了，不代表 GPU 已经读完那块内存。

第一步不是写 pool，而是补计数。热身后的五次正式运行显示：

```text
Qwen generate       12,345 次 allocate/free
DeepSeek generate   53,865 次 allocate/free
Qwen train           9,200 次 allocate/free
DeepSeek train      10,715 次 allocate/free
```

机会是真实的。保留版使用 exact-size block 和 HIP Event：析构时在默认 Stream 后面
放一个完成标记，只有标记 ready 才能复用。只要程序创建 `Stream` 或传入 external
stream，pool 就永久关闭并回到普通 `hipMalloc/hipFree`。这很保守，但不会拿异步正确性
换速度。

第一版仍然失败了：它从程序启动就缓存，连权重加载的巨大临时块也不释放，导致
Qwen/DeepSeek inference reserved memory 约变成 3.96/14.29 GB。最终版在 warm-up
同步完成后才显式启用，只学习 steady-state 尺寸。

| Workload | Step 05 | Step 06 | 本步加速 | 当前 PyTorch ratio |
|---|---:|---:|---:|---:|
| Qwen train | 72.33 | 107.08 token/s | 1.48× | 2.0864 |
| Qwen generate | 93.34 | 134.87 token/s | 1.45× | 1.9217 |
| DeepSeek train | 49.47 | 69.77 token/s | 1.41× | 2.6603 |
| DeepSeek generate | 38.99 | 48.93 token/s | 1.25× | 0.7842 |

```text
score                  1.219170 → 1.700597
Qwen gen backend alloc    12,345 → 305
DeepSeek gen backend alloc 53,865 → 810
```

这里也有反方向证据：rocprof 插桩下单次 Qwen decode 从 50.04 降到 46.89 token/s，
因为它放大了数千次 Event API 的开销。固定的未插桩主指标明显提升，所以方案保留；
但 Event batching 被记录成下一版问题，不能删掉这条不漂亮的数据。

## 17. Experiment 007：命中缓存不等于端到端更快

最新 trace 中 projection GEMM 占比最高，于是候选缓存了 hipBLASLt operation
descriptor 和三个 matrix layout。测试结果很漂亮：同一个 key 第一次 miss，之后 hit；
NN/NT/TN/TT 和 dtype 不会串 key，数值也完全通过。

但固定矩阵不接受“代码看起来合理”：

```text
running best score      1.700597
candidate score         1.669755
Qwen generation           -6.1%
DeepSeek training          -5.2%
```

一个可能解释是 descriptor 创建原本就不是主要成本；另一个解释是长期保留 host 对象
改变了 allocator/Event 的微小时序。当前证据不能区分，也没有必要靠猜测保留代码。
候选被删除，raw JSONL 留下，进度图出现第一个灰点。

这正是 autoresearch 风格记录的意义：running best 不下降，失败实验也不会消失。

## 18. Experiment 008：一次快、一次慢时怎么办

第二个候选只缓存 hipBLASLt 官方声明可序列化的 algorithm，不缓存 descriptor。
Focused 测试仍全部通过。第一次结果低于 running best，但同一时段重新运行原版后，
候选看起来又更快。

这不是可以挑数据的理由，而是测量协议不够强的信号。我们让 baseline 和 candidate
各运行三个独立进程，每个进程内部仍是 2 次热身、5 次正式计时，再逐 workload 取
中位数：

| Workload | Baseline median | Candidate median | Change |
|---|---:|---:|---:|
| Qwen train | 107.08 | 107.39 | +0.3% |
| Qwen generate | 134.87 | 122.58 | -9.1% |
| DeepSeek train | 68.77 | 67.31 | -2.1% |
| DeepSeek generate | 49.05 | 48.94 | -0.2% |

候选被 discard。比代码更重要的产物是新规则：从此声称小于 10% 的收益，至少需要
三次 baseline 和三次 candidate 进程中位数。单次高点不能决定合入。

## 19. Experiment 009：融合也要保留短序列失败

cached Attention 原来分三步：写 scores、softmax、再乘 V。每步之间都有 Tensor、
allocator Event 和 Kernel launch。新路径让一个 block 负责一个 query head，概率只放
shared memory；sequence 超过 4096 时仍回退旧实现。

由于预期收益不大，这次直接使用三进程中位数。训练代码没改，所以 score 复用
baseline 训练中位数，不把时间漂移算成 Attention 功劳：

```text
Qwen generation      134.87 → 142.25 token/s  +5.5%
DeepSeek generation   49.05 → 53.04 token/s   +8.1%
robust score         1.695566 → 1.752183
```

交错 context 曲线比短 benchmark 更有意思：

```text
1 token      -7.8%
32 tokens   +18.5%
128 tokens  +18.5%
512 tokens  +57.9%
```

融合并非到处更快。一个 token 的工作太少，shared-memory 和融合 Kernel 启动成本反而
更高。这个失败没有被条件分支“修饰掉”；当前只声明 32–512 点的收益，prefill、
backward、BF16 也仍未完成。

## 20. Experiment 010：正确的原地操作也可能一次都用不上

梯度累加候选只有在 Storage 唯一时才 `add_`，否则继续旧的 out-of-place add。别名、
重复 backward 和完整 Transformer 都正确。

但 measured allocation 给出了最干净的反驳：

```text
Qwen train      9,200 → 9,200
DeepSeek train 10,715 → 10,715
```

原因是梯度累加时通常仍与上游节点共享 Storage，`use_count()==1` 很少成立。一次候选
训练恰好较快不能覆盖“零分配减少”这个主指标。代码被删除；下一版若继续，必须显式
计算每个节点还剩几个梯度贡献，才能安全转移所有权。

## 21. Experiment 011：少一个 Kernel 也可能更慢

Q/K/V projection 原来是 GEMM 后单独 add bias。hipBLASLt bias epilogue 可以在 GEMM
内部完成，候选也真的减少了分配：Qwen 11,145→9,345，DeepSeek 48,545→40,565。

但三进程中位数是：

```text
Qwen generation      -7.8%
DeepSeek generation  +2.1%
candidate score       1.725932 < 1.752183
```

Qwen 的融合 GEMM 路径显然付出了更大代价。这个实验再次说明不能用 Kernel 数量替代
端到端测量。代码和 API 已删除，第四个灰点保留。

## 22. Experiment 012：Kernel 快 30 倍，整机只快一点

151,936 词表的 argmax 原来只有一个 block，每个线程要检查约 594 个数。两阶段版本
让最多 256 个 block 先找局部最大，再用一个 block 汇总。

```text
argmax Kernel    2.043 ms → 0.067 ms  (-96.7%)
Qwen median      +3.6%
DeepSeek median  +0.6%
robust score     1.752183 → 1.770568
```

这是一堂很典型的系统课：Kernel 快约 30 倍，不代表模型快 30 倍，因为它原本只占
一小部分时间。rocprof 插桩下整机甚至反向变慢，说明两次 Kernel 和 scratch/Event
更容易被工具放大。代码保留，但结论只写“边缘端到端收益”。

## 23. Experiment 013：官方有 API，不代表当前 shape 有 Kernel

hipBLASLt extension 提供 GroupedGemm，看起来正适合 Q/K/V：三个权重、同一个输入、
三个输出，而且不需要复制权重。

实际 MI300X 探针：

```text
M=1 K=128 N={128,64,64}    no heuristic
M=1 K=128 N={128,128,128}  no heuristic
```

变宽和同宽控制都 fallback。继续跑模型只能测三次旧 GEMM，所以实验在 operator gate
就停止，API 和代码删除。以后可以在 BF16 或 prefill 的较大 M 重新问这个问题。

## 24. Experiment 014：BF16 不是一个全局开关

BF16 track 先实现 device-native cast 和“BF16 输入、FP32 累加/输出”GEMM。它没有
改 FP32 running-best，也没有提前声称整网 BF16。

五个 M=1 shape 把问题说得很清楚：

![BF16 mixed GEMM shape track](assets/bf16-gemm.svg)

```text
1×384×384    0.87× FP32
1×896×896    1.07×
1×896×4864   0.83×
1×1536×1536  0.95×
1×1536×8960  1.15×
```

所有数字都包含 activation cast，输出保持 FP32。两个 shape 加速，三个 shape 退化，
所以当时的下一步是验证 per-shape policy 和 cached BF16 weights，而不是直接加一个
“全部 BF16”按钮。Experiment 015 随后实测并否决了这条候选。BF16 结果单独画图，
不能接到 FP32 1.770568 曲线上。

## 25. Experiment 015：算子快，不等于把它塞进模型就会快

Experiment 014 只有两个 shape 胜过 FP32。候选因此没有打开“全局 BF16”，而是只对
`896×896` 和 `1536×8960` 做精确 allow-list，并缓存对应 BF16 权重。

结果仍然失败：

![BF16 model policy was rejected](assets/bf16-model-policy.svg)

```text
Qwen median       147.41 → 125.29 token/s  -15.0%
DeepSeek median    53.36 →  51.85 token/s   -2.8%
extra memory       +73.5 MiB / +1.44 GiB
generated tokens   exact match
```

为什么“小算子加速”没有兑现？当前设计每次 Linear 仍要把 FP32 activation 转成 BF16，
并长期保存一份额外的 BF16 权重。Kernel 选择、cast 和内存代价合起来超过了两个 shape
的局部收益。因此模型 precision enum、cache 和 CLI 全部删除。

保留下来的是更小、证据更完整的能力：`bf16_matmul` 自动求导原语。前向对两个输入按
BF16 舍入并以 FP32 累加，反向使用 FP32 master tensors 计算梯度。它通过独立 PyTorch
前向/反向 oracle，但这只是一块训练地基，绝不等于“整网 BF16 已跑通”。

## 26. Experiment 016：融合连续的两个小算子

最新 DeepSeek trace 里，bias 和 split-half RoPE 分别占 4.9% 和 2.9% Kernel 时间。
Q/K projection 总是先加 bias 再做 RoPE，所以候选让一个 Kernel 直接读取原 projection
与 head-aware bias，并写出旋转后的结果；V bias 仍走原路径。

这次没有只比较历史单点，而是在同一时段独立构建旧 commit，让 baseline/candidate
各跑三进程中位数：

```text
Qwen train          112.32 → 112.43 token/s   +0.1%
Qwen generate       124.88 → 142.01 token/s  +13.7%
DeepSeek train       67.15 →  67.41 token/s   +0.4%
DeepSeek generate    52.05 →  55.50 token/s   +6.6%
fixed-reference score         1.770568 → 1.784147
```

Profiler 的因果链也吻合：总 Kernel 调用 11,804→10,684，正好少 1,120 次；add_bias
1,680→560，旧 RoPE 1,120→0，新 fused Kernel 1,120 次。`hipLaunchKernel` API 时间
77.80→65.26 ms，Qwen/DeepSeek inference allocation 分别减少 1,200/5,320 次。

融合操作还有独立 CPU reference、HIP 对照与 PyTorch 前向/输入梯度/bias 梯度 oracle，
不是只在模型里藏一个未经验证的 Kernel。候选保留，但 DeepSeek generation 仍只有
PyTorch 的 0.889×，下一轮还要继续处理剩余 GEMM 和 launch 结构。

## 27. Experiment 017：综合分变好，也要写下单项退化

Block 的第一条 residual 后面立刻接 FFN RMSNorm，但 residual 本身还要给第二次 add。
新 Kernel 因此同时输出 `{sum, normalized}`，每层少一个 launch，训练图保持原样。

```text
Qwen generation      +8.9%
DeepSeek generation  -4.2%
score                 1.784147 → 1.803226
Kernel calls          10,684 → 10,152
```

这次最值得保留的不是更高总分，而是矛盾证据：rocprof 插桩下 DeepSeek 从 28.28 提高到
29.74 token/s，三进程未插桩中位数却下降 4.2%。候选仍满足“任何单项不得退化超过 5%”
且总分提高，因此保留；但报告明确写出 DeepSeek 退化，下一轮不能把它当作已解决。

## 28. Experiment 018：用失败找到真正需要调的 shape

节点 017 的 DeepSeek 退化留下了一个具体问题：同一个 fused Norm Kernel 同时服务
896 和 1536 两种 hidden width。候选只让 width≥1024 使用 512 threads，896 路径不动。

```text
DeepSeek generation   53.20 → 58.32 token/s  +9.6%
target Kernel          6.46 →  4.83 μs       -25%
score                  1.803226 → 1.845199
```

这不是“512 总比 256 好”。目前只测了两种官方 hidden width，1024 边界只是一条有证据
的两区策略。更多 width 没测之前，不能把它包装成通用 autotuner。

## 29. Experiment 019：少一些空闲线程，反而更慢

cached Attention 固定用 256 threads，而两个模型 head width 只有 64/128。把 block 缩成
64/128 看起来很自然，但三进程中位数直接否决了它：

```text
Qwen generation      -6.6%
DeepSeek generation  -4.9%
score                 1.845199 → 1.791371
```

较大 block 可能在隐藏串行 dot-product、内存访问或 occupancy 上更有利。候选代码删除，
灰点保留。这再次说明“线程利用率看起来更高”不能替代端到端测量。

## 30. Experiment 020：全算法搜索也要过模型门

`hipblaslt-bench --algo_method all` 为 DeepSeek 的 `1×1536×1536` 找到了 explicit
solution。初次结果看起来非常快，但 100 次稳定复测只有：

```text
explicit solution    9.50 μs
default heuristic    9.87 μs
DeepSeek median      58.32 → 56.39 token/s  -3.3%
```

Kernel 确实换了，数值也正确，但端到端没有兑现。版本/shape 硬编码全部删除。这个灰点
比“调优成功”更重要：即使使用官方全搜索，仍不能跳过重复模型测量。

## 31. Experiment 021：API 调用少了，为什么还要拒绝

thread-local device cache 把 DeepSeek trace 的 `hipSetDevice` 从 30,669 次降到 1 次，
插桩 decode 也从 29.27 提高到 31.07 token/s。但未插桩四项全部下降：

```text
Qwen generate      -8.6%
DeepSeek generate  -5.2%
Qwen train         -4.9%
DeepSeek train     -1.7%
```

此外，外部库若直接改变当前 HIP device，thread-local 假设可能过期。API 数量和 profiler
都不能覆盖真实矩阵与互操作风险，所以代码删除。

## 32. Experiment 022：八块内存共用一个完成标记

exact-size pool 原来每释放一个 Tensor 就创建并记录一个 Event。新设计把默认 Stream 上
连续退休的八块内存放在同一完成边界后面；只有共享 Event ready，任何一块才能复用。

```text
Event record calls      8,993 → 1,124
Event record API time   24.39 → 1.95 ms
Qwen train/gen ratio    3.016× / 2.869× PyTorch
DeepSeek train/gen      3.126× / 1.206× PyTorch
score                   1.845199 → 2.389841
```

Kernel launch 数完全没变，说明收益来自 allocator 生命周期开销，而不是偷偷换了计算。
不满八块的尾批次会在显式 synchronize 前提交；external Stream 仍永久禁用 pool。

## 33. Experiment 023：批次继续变大，必须重跑四项

把 Event batch 从 8 调到 16 后，Event record 1,124→562。未插桩三进程结果是：

```text
Qwen train/generate      +2.6% / +8.4%
DeepSeek train/generate  -1.1% / +3.8%
score                    2.389841 → 2.470863
```

插桩 DeepSeek 反而变慢，backend allocation 也有小幅增加；这些代价都写进报告。16 被
保留是因为固定矩阵通过，而不是因为“batch 越大越好”。

## 34. Experiment 024：Event 不是越少越好

batch 32 虽然继续减少 Event，却让 block 等待更久：Qwen generation -4.8%，score
`2.470863 → 2.462231`，backend allocation 也增加。候选删除，16 成为当前测量矩阵的
局部最优，而不是拍脑袋常量。

## 35. Experiment 025：小收益也要三进程

K/V cache 原来分两次写同一 position。合并后 Qwen/DeepSeek generation 中位数分别
+0.4%/+0.8%，score `2.470863 → 2.478439`。收益很小，所以只有三进程都完成、两项都
不退化且直接测试通过后才保留。

## 36. Experiment 026：少一次 allocation 仍可能更慢

V bias 融入 cache store 后，Qwen/DeepSeek allocation 少 600/2660 次，但 generation
中位数下降 4.4%/1.0%。候选删除，说明 allocation 数也不能脱离端到端结果解释。

## 37. Experiment 027：shared memory 不是免费缓存

query 预载入 shared memory 后，Qwen +0.18%，DeepSeek -1.63%，score 下降。短 sequence
下 global query 已被硬件 cache 命中，额外 copy/sync 反而不值，候选删除。

## 38. Experiment 028：补完中间点再说局部最优

Event batch 24 的首轮四项全部低于 16，score 仅 `2.430451`。结合 8/16/24/32 四点，
当前矩阵的局部最优才可以写成 16，而不是只凭两个端点猜测。

## 39. 怎样读进度图

图中：

- 灰色点：完整跑过但 discard；
- 红色叉：crash/invalid；
- 绿色点：keep；
- 绿色阶梯：running best；
- 虚线：固定 PyTorch parity 1.0；
- 右侧条形：当前四项 workload ratio；
- 底部卡片：计划步骤，不代表已经完成。

FP32 主图当前有 baseline 和十四个 keep 实验共十五个绿色点，以及十二个 discard 灰点；
BF16 独立图另有一个被否决的模型策略。未来如果十个实验都失败，图上就应出现十个
灰点，而不是凭空出现一条漂亮上升曲线。

## 40. 什么才算从 0 到 1

完成一个 Kernel 不是 1，某个 shape 跑得快也不是 1。

本专项的“1”需要：

1. 固定 Qwen/DeepSeek train/generate 四项矩阵达到既定门；
2. exact tokens、loss、gradients 和 updates 仍通过；
3. 显存口径诚实；
4. 优化前后 raw JSONL 和 profiler 可追溯；
5. discard/crash 没有被删除；
6. 新学习者能沿日志重放关键实验；
7. 所有结论都写清适用 GPU、dtype、shape 和版本。

下一篇更新继续 Step 08 的 prefill/backward，或依据新 trace 选择更高收益热点。
