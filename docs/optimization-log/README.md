# microLLM-rocm optimization log

这是一个独立、可追溯的单卡性能优化实验室。它记录 microLLM-rocm 怎样从当前
MI300X FP32 基线逐步逼近固定 Python/PyTorch ROCm 参考，而不是在改完代码后只写
一篇成功总结。

![microLLM optimization progress](assets/progress.svg)

这张图借鉴了 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的
表达方法：灰点表示没有保留的实验，绿点表示保留的改进，阶梯线表示 running
best。图表、布局和数据均为本项目重新实现；没有复制原仓库的图片。

## 目标

baseline 四项固定 workload 的 microLLM/PyTorch throughput 比值为：

```text
Qwen2.5-0.5B train                  0.1422
Qwen2.5-0.5B generate               0.2675
DeepSeek-R1-Distill-Qwen-1.5B train 0.2209
DeepSeek-R1-Distill-Qwen-1.5B generate 0.1606
```

专项主分数是四个比值的几何平均：

```text
score = geometric_mean(workload throughput ratios)
baseline = 0.191660
current running best = 2.478439
selected-matrix parity target = 1.000000
```

Experiment 001 并行化 CrossEntropy；Experiment 002 让 GEMM 直接理解“转置读取”；
Experiment 003 并行化 RMSNorm；Experiment 004 将 KV Cache 留在 GPU 并直接映射 GQA。
Experiment 005 再让 greedy argmax 留在 GPU。当前 Qwen train/generate ratio 为
Experiment 006 只在 steady-state 默认 Stream 上启用 exact-size pool。当前 Qwen
train/generate ratio 为 `2.086361/1.921682`，DeepSeek train/generate ratio 为
`2.660319/0.784152`。
训练比值超过 1 只适用于当前极短 context 的固定 FP32 测量，不能推广成完整训练领先。
综合分数超过 1 也不表示每一项都达到 parity；DeepSeek generation 仍是明显缺口。

Experiment 007 的 hipBLASLt descriptor/layout cache 数值正确但被 discard：候选分数
`1.669755` 低于 running best，Qwen generation 和 DeepSeek training 分别退化约
6.1%/5.2%。灰点会保留在图上。

Experiment 008 只缓存可序列化 algorithm，同样被 discard。它还暴露了进程间波动：
单次比较会给出相反解释，三进程中位数最终确认 Qwen generation 退化 9.1%。此后
小于 10% 的候选强制使用 baseline/candidate 各三进程中位数。

Experiment 009 融合 cached decode 的 score 使用新中位数协议：Qwen/DeepSeek
generation 分别为 `2.026893×/0.849978×` PyTorch；训练路径未修改，因此复用 baseline
训练中位数，不把时间漂移冒充成 Attention 收益。

Experiment 010 的 copy-on-write gradient `add_` focused correctness 通过，但 Qwen 和
DeepSeek measured allocations 仍精确为 9,200/10,715，因主假设失败而 discard；没有
用一次较快训练进程冒充收益。

Experiment 011 的 hipBLASLt bias epilogue 确实减少 16–17% generation allocations，
但 Qwen generation 中位数退化 7.8%，因此同样 discard。少一个 Kernel 不等于更快。

Experiment 012 将大词表 argmax 改成两阶段 reduction，Kernel time 降低 96.7%。三进程
Qwen/DeepSeek generation 中位数再提高 3.6%/0.6%，当前 robust score 为 `1.770568`。

Experiment 013 探测无权重复制的 GroupedGemm QKV：变宽和同宽 FP32 M=1 两种控制都
没有可用 heuristic，因此直接 discard，没有拿三 GEMM fallback 冒充 grouped 加速。

Experiment 014/015 建立独立 BF16 shape 与 autograd track。精确 shape allow-list 在
Qwen/DeepSeek 官方模型上仍分别退化 15.0%/2.8%，并增加 73.5 MiB/1.44 GiB engine
memory，所以模型策略和 CLI 被删除；经过 PyTorch 前向/反向对齐的 BF16 自动求导
原语保留。这个结果不进入 FP32 running-best 曲线。

Experiment 016 将 Q/K projection bias 与 split-half RoPE 合并。配对三进程中位数中，
Qwen/DeepSeek generation 分别提高 13.7%/6.6%，两项训练变化为 +0.1%/+0.4%；
当前 FP32 running-best score 为 `1.784147`。完整前向、反向、AdamW 和 token 门通过。

Experiment 017 在 cached inference 中同时输出 residual sum 与 RMSNorm。Qwen generation
中位数提高 8.9%，DeepSeek 退化 4.2%，综合 score 提高到 `1.803226`。DeepSeek 反例和
profiler 中相反的 +5.2% 插桩结果都保留，没有用平均数掩盖单项。

Experiment 018 只将 hidden width 至少 1024 的 fused residual-Norm 改为 512 threads。
DeepSeek generation 三进程中位数提高 9.6%，目标 Kernel 平均时间降低约 25%，恢复并
超过 Experiment 016 的 DeepSeek 结果；当前 score 为 `1.845199`。

Experiment 019 按 head width 将 cached Attention block 缩为 64/128 threads，但 Qwen 与
DeepSeek generation 中位数分别退化 6.6%/4.9%，score 降到 `1.791371`。候选删除，
失败数据进入灰点。

Experiment 020 使用官方 hipBLASLt 全算法搜索得到的 exact-shape solution。稳定复测只
比默认 heuristic 快约 3.7%，DeepSeek 中位数反而退化 3.3%，因此版本/shape 硬编码被
删除。它成为“GEMM 局部更快但模型不快”的直接反驳实验。

Experiment 021 将重复 `hipSetDevice` 从 30,669 次降到 1 次，插桩运行也变快；但固定
未插桩矩阵四项全退化，两项生成超过 5% 拒绝门。thread-local 设备缓存删除，避免
破坏外部 HIP 调用的设备状态。

Experiment 022 让八个默认 Stream retired blocks 共用一个完成 Event。Event record 调用
约降 8×，四项 workload 中位数全部提高，DeepSeek generation 首次超过固定 PyTorch
参考；当前 score 为 `2.389841`。

Experiment 023 将批次从 8 调到 16。Event 再减半，三项 workload 改善、DeepSeek train
退化 1.1%，score 提高到 `2.470863`；显存峰值不变。

Experiment 024 继续增到 32，但 score 降到 `2.462231`，Qwen generation 退化 4.8%，
backend allocation 增加。代码恢复到实测局部最优 16。

Experiment 025 将同 position 的 K/V cache store 合成一次 launch。两模型生成中位数均
小幅提高，score 达到 `2.478439`。

Experiment 026 继续把 V bias 融入 store，allocation 明显减少但两模型中位数均退化，
因此 API/Kernel/模型改动全部删除。

Experiment 027 将 cached Attention query 放入 shared memory，Qwen 微增但 DeepSeek 和
score 退化，候选删除。

Experiment 028 补测 Event batch 24，四项全部低于 batch 16。至此 8/16/24/32 容量搜索
闭合，16 保持局部最优。

Experiment 029 跨 Block 融合 residual+Norm，少 28 个 launch 但 Qwen 退化 4.4%，调度
重排删除。

Experiment 030 没有重试被否决的“每层 cast + 双份权重”方案，而是先建立连续 BF16 FFN
激活岛。固定 Qwen/DeepSeek、M=1/128 四组 shape 中，相对 FP32 为 `1.117×–1.576×`，
相对逐 Linear BF16 为 `1.067×–1.091×`。真实 Qwen decode 还暴露并修复了小 M 直接
BF16→FP32 输出失败。它仍是算子 track，不是整网 BF16 声明。

![BF16 FFN activation island](assets/bf16-ffn-island.svg)

Experiment 031 将官方模型的 FFN 权重事务式替换成单份 BF16。Qwen/DeepSeek 相对本项目
FP32 decode 提高 `11.5%/5.1%`，常驻 engine memory 降约 32%，exact token 全通过；但
对 PyTorch 全 BF16 的四项 prefill/decode 只有 Qwen decode 超过 1.0，所以状态是 partial
keep，不是“全面超越 PyTorch”。

![Official-model BF16 FFN inference](assets/bf16-model-inference.svg)

Experiment 032 发现 prefill 测量前没有启用已保留的 allocator。修正生命周期边界后，
Qwen/DeepSeek BF16 prefill 分别提高 `1.642×/1.535×`，decode 不退化。相对 PyTorch
BF16 的四项现在三项过线，只剩 DeepSeek decode。

![Prefill allocator before and after](assets/bf16-prefill-allocator.svg)

Experiment 033 只 profile 剩余的 DeepSeek decode 红条。GEMM 占 Kernel 时间 67.64%，
并且调用数可还原为每层 7 个 Linear 加 tied output head；当前只有 3 个 FFN Linear
进入 BF16。下一步边界因此是 Attention Linear 的单份 BF16，而不是继续改 prefill。

Experiment 034 先保留“每个 Q/K/V 各 cast 一次”造成的回退，再加入共享 input cast。
三进程 Qwen decode/prefill 提高 `2.9%/6.9%`，DeepSeek decode 提高 2.0%、prefill
退化 2.7%（未越 5% 门），常驻权重继续下降；DeepSeek decode 仍只有 PyTorch BF16
的 `0.533×`。

![BF16 Attention shared cast](assets/bf16-attention.svg)

Experiment 035 复测 retained Attention 后，GEMM Kernel 时间已经下降，但 3,743 次 BF16
GEMM 仍在 host 重建 description/layout。Experiment 036 只为这条 BF16 path 增加
thread-local immutable exact-shape plan；Qwen decode/prefill 提高 `2.93×/2.74×`，
DeepSeek提高 `2.55×/2.67×`，固定 PyTorch BF16 四项全部过线。

![BF16 immutable plan cache](assets/bf16-plan-cache.svg)

Experiment 037 建立 BF16 Linear + FP32 master/gradient/AdamW 训练。两模型吞吐是 PyTorch
BF16 autocast 的 `3.12×/2.58×`，loss 都下降；但相对 microLLM FP32 仅
`0.918×/0.906×`，峰值显存完全不降，所以只保留训练地基，不宣称内部加速。

![BF16 FP32-master training](assets/bf16-training.svg)

Experiment 038 对照 trace 证明 BF16 GEMM 快 20.9%，但 360 个 cast 的 1.91 ms 超过
GEMM 节省的 1.33 ms。Experiment 039 让训练 Q/K/V 共享 activation cast，allocation
精确少 240/280 次，却使两模型几何吞吐变成约 `0.991×`；候选 graph API 删除，失败图保留。

![BF16 training shared QKV discarded](assets/bf16-training-qkv-discard.svg)

Experiment 040 不再每次 forward 重建 BF16 Linear 权重。每个 FP32 master 持有一个
持久 BF16 镜像，AdamW 在同一个 HIP launch 中同时写回 master 和镜像；checkpoint 只存
主状态，恢复 optimizer 时重建派生镜像。Qwen/DeepSeek 相对 Experiment 037 提高
`9.4%/5.9%`，但峰值显存增加 `7.9%/10.8%`，因此作为显式速度/显存选项保留。

![Persistent BF16 training mirrors](assets/bf16-training-mirrors.svg)

Experiment 041 尝试连续 BF16 FFN 训练岛。完整 tiny Transformer logits/loss/全部梯度
与 PyTorch 对齐，Qwen 五步净少 120 次 allocation；但同一性能漂移窗口内只从
`18.685→18.892 token/s`，即 `1.011×`，未过 5% 门。最初拿旧的 151.69 基线会误判
为 8× 回退，同窗口 control 推翻了这个解释。候选代码删除，测量失败和早停记录保留。

![BF16 FFN training island discarded](assets/bf16-training-ffn-island-discard.svg)

Experiment 042 建立新的官方训练 shape 基线。Qwen BF16 在 `1×3、2×3、1×32、1×128`
上分别达到 PyTorch BF16 autocast 的 `0.413×、0.341×、0.131×、0.352×`。24 条 raw
全部更新参数且 token 计数正确。context 32 反而明显慢于 128，说明下一步必须 profile
shape-specific GEMM/调度，不能只用 Attention 复杂度解释。

![Official training shape baseline](assets/bf16-training-shape-matrix.svg)

Experiment 043 用 profiler 证明 context 32 的主要问题是 507 次 readable transpose
weight-gradient GEMM，占 Kernel 时间 75.75%。精确 shape micro-benchmark 中 hipBLASLt
快 `1.54×–21.99×`。只扩展这类宽输出 `transpose(left)` 的 Auto 路由后，四个 Qwen
shape 分别提高 `1.659×、2.020×、4.476×、1.007×`，显存不变，候选保留。

![Weight-gradient routing result](assets/bf16-weight-gradient-routing.svg)

Experiment 044 将 full-sequence QK、causal softmax、PV 与 GQA head mapping 合成一个
前向/反向边界，不再保存 T×T scores/probabilities 或复制 K/V heads。四个 Qwen shape
再提高 `1.052×–1.218×`，context 128 峰值少 185.6 MB；完整 PyTorch/CPU/HIP 梯度与
trace 对齐通过。

![Fused causal GQA training](assets/fused-causal-gqa-training.svg)

Experiment 045 将同一训练矩阵扩展到 DeepSeek 1.5B，并修复官方权重加载架构。未初始化
模型不再生成/复制即将覆盖的 1.78B 随机参数；safetensors 直接进入设备，Linear transpose
在 GPU 完成。观察到的进程准备从约 6–7 分钟降到约 80 秒，正式 `load_ms` 约 65 秒。
DeepSeek 四 shape 达到 PyTorch 的 `0.337×–0.532×`，峰值显存低 `8%–12%`。

![DeepSeek training shapes and load time](assets/deepseek-training-shapes.svg)

Experiment 046 在 DeepSeek `1×128` 上保留一次完整进程 profile。AdamW 的 1,017 次
launch 精确对应 339 个参数 × 3 个 step，占 Kernel 时间 `32.94%`，成为最干净的下一
训练热点。`strided_copy` 的 `23.00%` 混有加载期 GPU transpose，不能全部归因于训练。
因此下一节点先稳定梯度地址，再构建持久 pointer table 和 multi-tensor AdamW。

![DeepSeek context-128 optimizer profile](assets/deepseek-context128-profile.svg)

Experiment 047 测试了 multi-tensor 的第一个前置方案：跨 `zero_grad()` 保留叶子梯度
Storage。地址、数值和零 payload transfer 全部通过，但首贡献 copy 让 Qwen `1×128`
匹配协议中位数从 `802.70` 降到 `757.48 token/s`（`−5.63%`）。峰值少 544.5 MB 仍不足以
越过吞吐拒绝线；代码删除，下一版改用 16 个 tensor 一组的 Kernel 参数描述符。

![Stable gradient buffer discard](assets/stable-gradient-buffer-discard.svg)

Experiment 048 改成 launch 时传入当前指针。全量 16-Tensor 分组把 Qwen AdamW
`290→19` 次，却让 `1×128` 吞吐降到 `0.577×`，立即早停。只融合不超过 4,096 元素的
121 个小 Tensor 后，dispatch 为 177 次；四 shape speedup 为
`0.988×、1.027×、1.022×、1.005×`，显存不变。没有一项过 5%，代码仍然删除。

![Chunked AdamW discard](assets/chunked-adamw-discard.svg)

Experiment 049 改测单个大 Tensor 的数据通路。float4 在带 BF16 mirror 的部分 exact shape
达到 `1.056×–1.194×`，但 width 8、rsqrt、无 mirror 大权重都有反例；强制全模型 Vectorized
的四个 Qwen pilot 为 `0.965×–0.994×`。因此实现和独立 benchmark 作为显式研究路径保留，
`Auto` 仍选择 Scalar，不把局部算子收益冒充模型加速。

![Vectorized AdamW explicit policy](assets/vectorized-adamw-explicit.svg)

Experiment 050 把未初始化 HIP 模型的单文件 safetensors 改为 header 预检 + payload 顺序
streaming + 可复用低精度 staging + cast/transpose out。Qwen load `17.659→0.580s`
（`30.45×`），DeepSeek `65.100→1.356s`（`48.02×`），H2D 正好等于 BF16 文件字节。
DeepSeek 四 shape 训练吞吐变化在 `−0.4%–+0.1%`，峰值不变，候选保留。

![Streaming safetensors load](assets/streaming-safetensors-load.svg)

Experiment 051 将正式训练扩展到 context 512。Qwen/DeepSeek 数值与参数更新通过，但吞吐仅
为 PyTorch 的 `0.0978×/0.0832×`，峰值为 `1.239×/1.033×`。Qwen retained profile 中
causal GQA backward/forward 占 Kernel 时间 `50.64%/13.86%`，下一反驳实验锁定长序列
Attention backward 的原子 K/V 累加。

![Context-512 baseline and profile](assets/context512-training-profile.svg)

Experiment 052 把 Attention backward 拆成 row 矩阵生成和无原子 K/V reduction。T=256
Q/K/V 数值通过，但 Qwen T=512 吞吐 `812.45→688.82 tok/s`；backward Kernel 总时间
`985.61→1320.85ms`。输出线程重复扫描 query/head 和 T² 写读比 atomic 更贵，代码删除。

![Split K/V backward discarded](assets/split-kv-backward-discard.svg)

只提高平均数不够。每次保留改动还必须满足正确性、单项退化、显存和复杂度门。

## 目录

| 路径 | 用途 |
|---|---|
| [BLOG.zh-CN.md](BLOG.zh-CN.md) | 从 0 到 1 的完整优化博客，持续更新 |
| [PROGRAM.md](PROGRAM.md) | 人或 Coding Agent 执行实验时必须遵守的循环 |
| [PLAN.md](PLAN.md) | 阶段路线、优先级和完成定义 |
| [SCHEMA.md](SCHEMA.md) | 分数、状态、结果表和图表字段合同 |
| [SATURATION.md](SATURATION.md) | 已穷举的局部搜索空间与架构级下一阶段边界 |
| [results.tsv](results.tsv) | 一行一个实际实验；计划不进入结果表 |
| [steps/](steps/) | 每个优化步骤的假设、边界、测试和反驳实验 |
| [experiments/](experiments/) | 实际执行后新增的详细实验报告 |
| [assets/progress.svg](assets/progress.svg) | 从 `results.tsv` 生成的当前进度图 |
| [assets/bottleneck-map.svg](assets/bottleneck-map.svg) | 当前瓶颈和目标架构图 |
| [assets/bf16-gemm.svg](assets/bf16-gemm.svg) | BF16 mixed GEMM 独立 shape track |
| [bf16-results.tsv](bf16-results.tsv) | BF16 shape、速度、误差原始表 |
| [assets/bf16-model-policy.svg](assets/bf16-model-policy.svg) | 被否决的官方模型 BF16 策略图 |
| [bf16-model-policy.tsv](bf16-model-policy.tsv) | 三进程中位数、显存和 token gate |
| [assets/bf16-ffn-island.svg](assets/bf16-ffn-island.svg) | 连续 BF16 FFN 激活岛独立曲线 |
| [experiments/030-data/](experiments/030-data/) | 36 条 raw JSONL、摘要和 kernel trace |
| [assets/bf16-model-inference.svg](assets/bf16-model-inference.svg) | 官方模型 BF16 FFN 与两条 reference |
| [experiments/031-data/](experiments/031-data/) | 18 条 raw、准备峰值和聚合摘要 |
| [assets/bf16-prefill-allocator.svg](assets/bf16-prefill-allocator.svg) | prefill allocator 前后与 PyTorch 门 |
| [experiments/032-data/](experiments/032-data/) | 两模型/两策略三进程复测 |
| [experiments/033-data/](experiments/033-data/) | DeepSeek decode kernel/HIP API 聚合统计 |
| [assets/bf16-attention.svg](assets/bf16-attention.svg) | per-Linear cast 失败与 shared-cast 三进程结果 |
| [experiments/034-data/](experiments/034-data/) | 官方 logits/token、candidate raw 与 pilot |
| [experiments/035-data/](experiments/035-data/) | retained Attention 后的 profiler 聚合 |
| [assets/bf16-plan-cache.svg](assets/bf16-plan-cache.svg) | BF16 plan cache 与 PyTorch BF16 四项验收 |
| [experiments/036-data/](experiments/036-data/) | plan-cache 三进程 official raw/summary |
| [assets/bf16-training.svg](assets/bf16-training.svg) | FP32 master BF16 training 的成功与失败门 |
| [experiments/037-data/](experiments/037-data/) | 18 条 official train raw、摘要和 native-BF16 失败 |
| [experiments/038-data/](experiments/038-data/) | Qwen FP32/BF16 单步 profiler 对照 |
| [assets/bf16-training-qkv-discard.svg](assets/bf16-training-qkv-discard.svg) | 少 cast/allocation 但吞吐未改善 |
| [experiments/039-data/](experiments/039-data/) | shared-QKV candidate 三进程 raw/summary |
| [assets/bf16-training-mirrors.svg](assets/bf16-training-mirrors.svg) | 持久 BF16 权重镜像的吞吐/显存取舍 |
| [experiments/040-data/](experiments/040-data/) | 两模型三进程镜像训练 raw/summary |
| [assets/bf16-training-ffn-island-discard.svg](assets/bf16-training-ffn-island-discard.svg) | 同窗口 control 揭示共享 GPU 漂移与 1.1% 无效收益 |
| [experiments/041-data/](experiments/041-data/) | Qwen raw、DeepSeek early-stop 与 profiler 聚合 |
| [assets/bf16-training-shape-matrix.svg](assets/bf16-training-shape-matrix.svg) | Qwen batch/context 吞吐和显存曲线 |
| [experiments/042-data/](experiments/042-data/) | 四 shape、两框架、三进程的 24 条 raw |
| [assets/bf16-weight-gradient-routing.svg](assets/bf16-weight-gradient-routing.svg) | transpose weight-gradient 路由前后曲线 |
| [experiments/043-data/](experiments/043-data/) | 24 条候选 raw、microbench 与三组 profiler 聚合 |
| [assets/fused-causal-gqa-training.svg](assets/fused-causal-gqa-training.svg) | full-sequence Attention 融合前后吞吐/显存 |
| [experiments/044-data/](experiments/044-data/) | 24 条 raw、前后比较与 retained profiler |
| [assets/deepseek-training-shapes.svg](assets/deepseek-training-shapes.svg) | DeepSeek shape 与 load gap 曲线 |
| [experiments/045-data/](experiments/045-data/) | 优化前 pilot、24 条正式 raw 与 load 摘要 |
| [assets/deepseek-context128-profile.svg](assets/deepseek-context128-profile.svg) | DeepSeek context 128 训练热点与阶段污染边界 |
| [experiments/046-data/](experiments/046-data/) | retained Kernel/HIP API 聚合与可验证 profile 合同 |
| [assets/stable-gradient-buffer-discard.svg](assets/stable-gradient-buffer-discard.svg) | 稳定梯度地址的吞吐/显存反例 |
| [experiments/047-data/](experiments/047-data/) | 匹配协议 raw、错误协议保留与 discard 合同 |
| [assets/chunked-adamw-discard.svg](assets/chunked-adamw-discard.svg) | 全量/小 Tensor 分组与端到端反例 |
| [experiments/048-data/](experiments/048-data/) | 早停 pair、四 shape 24 条 raw 与 dispatch 合同 |
| [assets/vectorized-adamw-explicit.svg](assets/vectorized-adamw-explicit.svg) | exact-shape 算子收益与官方模型反例 |
| [experiments/049-data/](experiments/049-data/) | width4/8、sqrt/rsqrt、mirror/no-mirror 与 Qwen pilot raw |
| [assets/streaming-safetensors-load.svg](assets/streaming-safetensors-load.svg) | Qwen/DeepSeek load、H2D 和训练非退化 |
| [experiments/050-data/](experiments/050-data/) | load smoke、DeepSeek 24 条正式 raw 与安全合同 |
| [assets/context512-training-profile.svg](assets/context512-training-profile.svg) | T=512 PyTorch 比率、显存与 Kernel 类别 |
| [experiments/051-data/](experiments/051-data/) | pilot、12 条正式 raw 与 retained profiler 聚合 |
| [assets/split-kv-backward-discard.svg](assets/split-kv-backward-discard.svg) | atomic 基线与两阶段 K/V 反例 |
| [experiments/052-data/](experiments/052-data/) | pilot、candidate profiler 与 discard 合同 |
| [scripts/render_progress.py](scripts/render_progress.py) | 无第三方依赖的 SVG 生成器 |
| [scripts/validate_log.py](scripts/validate_log.py) | 日志、分数、链接和生成图一致性检查 |

## 固定实验循环

```text
读取 running best 和最新 trace
→ 只选择一个可反驳假设
→ 写实验合同
→ 建独立实验分支
→ 实现最小改动
→ 先跑正确性
→ 跑 2 warm-up + 5 measured
→ 与固定 PyTorch raw baseline 比较
→ 写 keep / discard / crash
→ 重新生成进度图
→ keep 才进入 main
```

与 autoresearch 不同，本项目不能只优化一个模型质量指标。系统优化必须同时守住：

- 完整 logits、loss、梯度和生成 token；
- train/generate 四项吞吐；
- engine/PyTorch allocator 峰值口径；
- CPU reference 和 readable HIP 路径；
- 单卡改变不能破坏多卡数值语义。

## 开始一次实验

复制模板：

```bash
cp docs/optimization-log/experiments/TEMPLATE.md \
  docs/optimization-log/experiments/NNN-short-name.md
```

按 [PROGRAM.md](PROGRAM.md) 执行。实验完成后向 `results.tsv` 追加一行，再生成图：

```bash
python3 docs/optimization-log/scripts/render_progress.py
python3 docs/optimization-log/scripts/validate_log.py
```

## 状态词

| 状态 | 含义 |
|---|---|
| `baseline` | 固定起点 |
| `keep` | 正确性通过，主指标改善，改动被保留 |
| `discard` | 实验完成但不满足保留门，代码不合入 |
| `crash` | OOM、编译失败、超时或运行错误 |
| `invalid` | workload、环境或证据发生变化，结果不可比较 |

计划中的想法只能写在 `steps/`，不能提前写进 `results.tsv`。
