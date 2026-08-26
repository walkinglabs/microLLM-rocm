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

> **Decode口径修正：** Experiment 085证明，历史cached generation计时把prefill免费给出的
> 第一个token计入了decode token数。它仍可用于同口径的microLLM候选前后筛选，但上面的
> generation/PyTorch比值不能再解释成steady decode排名。现在正式矩阵要求每个measured token
> 对应一次模型forward，并单独报告prefill准备时间。原始历史数据保留，不回填成新口径。
> Experiment 085的冻结Release N8矩阵显示：Qwen六个shape均达到或超过当前PyTorch参考；
> DeepSeek T8/T512过线，T2048 B1/B8仍为`0.866×/0.671×`，是新的明确优化目标。

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

Experiment 053 补上 strided-batched hipBLASLt。rank-4、FP32/BF16、四种 transpose 和零搬运
对齐通过；Qwen T=512 的 `QKᵀ` 与 `PV/Qgrad` exact shapes 相对 readable 分别快
`26.23×/113.64×`。transpose-left readable 计时受跨 Stream temporary 污染，明确排除。

![Strided-batched hipBLASLt](assets/strided-batched-hipblaslt.svg)

Experiment 054 用 row recompute + 两次 strided-batched GEMM + GQA head reduction 重写
T≥256 backward。Qwen/DeepSeek T=512 分别提高 `1.358×/1.365×`，measured peak 不变；
T=128 fallback `1.008×`。Qwen 总 Kernel 时间 `1.946→1.442s`，候选保留。

![Batched Attention backward](assets/batched-attention-backward.svg)

Experiment 055 仅在 Autograd T≥256 保存 forward causal probability，backward 不再重算
QK/softmax。Qwen/DeepSeek 再提高 `1.132×/1.150×`，代价是固定 +336 MiB measured peak；
T128 fallback `0.991×`。Qwen row backward `473.91→305.15ms`，候选作为显式长序列
speed/memory tradeoff 保留。

![Saved Attention probabilities](assets/saved-attention-probabilities.svg)

Experiment 056 把同一长序列 forward 的 `QKᵀ/PV` 接到 strided-batched hipBLASLt。
Qwen/DeepSeek T512 再提高 `1.091×/1.165×`，measured peak 不变；T128 fallback
`1.012×`。Qwen forward stage `272.52→178.29ms`，全 Kernel 时间
`1283.85→1185.53ms`。dispatch 增加 4.1%，因此保留依据是设备时间和端到端收益，
不是 launch 数。

![Batched Attention forward](assets/batched-attention-forward.svg)

Experiment 057 再把 saved backward 的 `dP/dQ` 接入 batched GEMM；`dK/dV` 已由前一
节点使用同一路径。Qwen/DeepSeek T512 提高 `1.201×/1.309×`，peak 仍不变；T128
单次 `0.987×`，在 5% 门内。旧 306.63ms saved-row Kernel 被 122.21ms 的 GEMM +
softmax backward stage 替代，全 Kernel 时间 `1185.53→988.36ms`。

![Fully batched Attention backward](assets/full-batched-attention-backward.svg)

Experiment 058 将 T≥256 causal softmax 的前向和反向从“一线程扫一行”改成“一 block
合作一行”。Qwen/DeepSeek T512 再提高 `1.302×/1.196×`，peak 不变；T128
`1.002×`。forward/backward softmax 分别 `4.253×/4.801×`，全 Kernel 时间
`988.36→772.84ms`，dispatch 数精确不变。

![Cooperative causal softmax](assets/block-row-causal-softmax.svg)

Experiment 059 将 rows≥256 RMSNorm weight gradient 从“一线程扫全部 rows”改为“一
block 合作一个 hidden column”。Qwen/DeepSeek T512 提高 `1.220×/1.125×`，peak
不变，T128 `1.003×`。目标 Kernel `142.77→8.72ms`（`16.38×`），全 Kernel
`772.84→646.97ms`，dispatch精确不变。

![Cooperative RMSNorm weight gradient](assets/block-column-rmsnorm-weight-gradient.svg)

Experiment 060 不优化Kernel，先尝试推翻旧推理结论。新runner把prefill、cache prepare、
steady decode、uncached decode分开，覆盖两模型、context 8–2048、batch 1–8、KV Storage/
active利用率和精度驻留策略。108条核心记录全通过且decode token一致；48条batch记录中
42条通过、6条microLLM cached B2/B4/B8明确unsupported。旧短prompt 4/4 parity不能推广：
T512 prefill仅为PyTorch的`0.044×/0.026×`，cached decode为`0.318×/0.267×`。

![Inference context, batch and KV matrix](assets/inference-context-batch-matrix.svg)

Experiment 061 先出现一个0.39%的“假优化”：operator有batched路由，但模型仍手写两次
readable matmul。rocprof显示144次占629.41ms。模型改为复用公共causal GQA后，
Qwen/DeepSeek T512/T1024提高`6.72×/13.18×`与`8.40×/16.73×`；T512 peak不变，
T1024增加33%/12%。全部top token一致，最大top-logit差0.195。

![Batched long-prefill inference](assets/batched-long-prefill-inference.svg)

Experiment 062 新增B1 full-sequence prefill-to-cache。第一次整块复制破坏capacity head stride，
继续decode测试失败；改为per-head D2D后通过。第二次返回完整`[T,V]`造成显存浪费，最终只
返回last logits。正式T1024 prepare为Qwen/DeepSeek `71/109ms`，不再是旧warm-up的
`38/55s`。同窗口Qwen T512 token/full profile：prepare `274.8×`、Kernel `112.3×`，
Kernel calls减少155×。

![Full-sequence prefill to KV cache](assets/full-prefill-kv-cache.svg)

Experiment 063 增加last-dim row-wise GPU argmax。Qwen/DeepSeek B1/2/4/8同卡host/device
全部变快`1.13×–2.15×`，peak/token不变。Qwen B8 measured D2H从38,895,616降到256B
（正好少151,936×），吞吐`115.2→252.0 tok/s`。profile中Kernel时间略增，端到端仍
2.06×，说明收益来自删除大传输/同步。

![Device row-wise argmax](assets/device-rowwise-argmax.svg)

Experiment 064 把KV Storage、prefix、step store、cached GQA和模型扩到batch维。Experiment060
的6条unsupported变成48条正式pass。Qwen B1→B8 `91.9→721.1 tok/s`、效率98.1%；
DeepSeek `62.2→494.6`、效率99.5%；micro/PT稳定约0.59–0.75。token/KV公式一致，
FP32 Cache字节仍为PyTorch BF16的2.057×。

![Batched KV cache](assets/batched-kv-cache.svg)

Experiment 065只把K/V Storage从FP32改成BF16，Attention仍FP32累加。匹配Release矩阵
72/72进程成功：Cache全部减半、16-token suffix 12/12一致、11/12 shape加速；DeepSeek
T2048 B8提高24.8%，peak降约5%。完整logits仍有一个失败：DeepSeek T512 B1 RMSE
`0.0586>0.05`，所以BF16 Cache保留为opt-in，FP32仍默认。

![BF16 KV cache](assets/bf16-kv-cache.svg)

Experiment 066尝试用一个prefix-pair Kernel删除BF16 prefill的96次cast和全部D2D。局部
profile更干净，但正式Qwen T2048 B8 prepare慢30.5%、端到端慢21.1%，三轮复现。
候选全部删除，只保留失败图和paired step-store的dtype合同修复。

![Fused prefix pair discarded](assets/fused-prefix-pair-discard.svg)

Experiment 067用逐层dtype修复uniform BF16唯一RMSE失败。仅layer 1 FP32就让完整logits
12/12通过，Cache仍缩小1.920×/1.931×。机制作为显式strict策略保留，不改变FP32默认；
早期跨时段性能代价由Experiment 069重新审查。

![Mixed-layer KV policy](assets/mixed-layer-kv-policy.svg)

Experiment 068只对strict策略的一个FP32层重试paired prefix。同binary D2D少160次/
167.8MB，但prepare/E2E仍慢1.53%/0.59%；候选删除，这一copy-fusion搜索空间关闭。

![Targeted prefix pair discarded](assets/targeted-prefix-pair-discard.svg)

Experiment 069用同binary、交替顺序重新配对uniform/strict。DeepSeek T2048 B8 E2E从旧的
跨窗口`0.866×`变为同窗口`1.011×`，推翻稳定长batch回退解释；新runner作为证据基础保留。

![Same-binary KV policy](assets/same-binary-kv-policy.svg)

Experiment 070换repeat/rotated/constant/ramp挑战strict。layer 1只过9/14，constant T512
RMSE达2.995并token分叉；前4层FP32策略14/14通过，Cache仍缩小1.75×，同binary性能最差
约0.97×。robust-strict配方据此更新。

![KV policy prompt robustness](assets/kv-policy-prompt-robustness.svg)

Experiment 071把同样挑战施加到Qwen。uniform BF16的constant三context全失败；first 2在
T512通过却在T2048 RMSE 3.141并token分叉，前4/8/12也无法修复，只有全FP32通过。

![Qwen KV prompt failure](assets/qwen-kv-prompt-failure.svg)

Experiment 072建立多请求serving reference：延迟到达、独立Cache/RNG、完成释放和CPU/HIP
对齐。1–8请求HIP吞吐约331 tok/s且不扩展，明确留下slot batching的before。

![Reference serving scheduler](assets/reference-serving-scheduler.svg)

Experiment 073新增静态跨请求`generate_batch()`。HIP B8相对serial为7.306×、扩展效率
90.7%，24/24进程逐row输出一致；等长/同配置限制和缺少slot refill明确保留。

![Static batch generation](assets/static-batch-generation.svg)

Experiment 074增加admission compatibility buckets、singleton fallback和跨drain到达。HIP B4
达到3.78×，B8/B16拆成多个B4组后吞吐平台约1260 tok/s，明确留下slot refill目标。

![Admission batch scheduler](assets/admission-batch-scheduler.svg)

Experiment 214沿Experiment 213的带宽合同把两个AdamW moment改成BF16。五进程正式门中，
Qwen/DeepSeek端到端提高`1.0226×/1.0356×`，峰值降到`0.8329×/0.8084×`，状态字节精确
减半。Qwen optimizer只有`1.0687×`，未过`1.10×` stretch gate，所以显式策略partial keep，
默认仍为FP32；全量multi-tensor反例也完整保留。

![BF16 AdamW moments](assets/bf16-adamw-moments.svg)

Experiment 215只合并小Tensor并扫描6个阈值。1M五进程正式门让Qwen/DeepSeek optimizer达到
`1.240×/1.263×`、端到端达到`1.049×/1.053×`；16M把DeepSeek降到
`0.896×/0.980×`。这条反例确定了“合并不是越多越好”，1M作为HIP BF16 moment Auto保留。

![Hybrid BF16 AdamW](assets/hybrid-bf16-adamw.svg)

Experiment 216重新采集Hybrid后的load-subtracted profile。AdamW相对Experiment 213提高
`1.372×/1.293×`，总Kernel提高`1.044×/1.058×`；GEMM现在占`59.33%/63.81%`。
AdamW阈值track关闭，下一训练节点必须改变GEMM结构。

![Post-hybrid training profile](assets/post-hybrid-training-profile.svg)

Experiment 217测试共享input的QKV/gate-up weight gradient。direct `N,T`列出8,153个算法，
共享转置后的`N,N`列出9,172个，但8个官方case全部0 supported。模型路由未创建，下一反驳
边界改为显式packed大GEMM。

![Grouped weight-gradient discarded](assets/grouped-weight-gradient-discard.svg)

Experiment 218再测试显式pack + 一个普通大GEMM。计时包含每步2/3次D2D pack，packed output可
直接作为参数gradient view，不隐藏split成本。四项数值误差≤1.15e-7，但速度只有
`0.835×–0.979×`，weight-gradient组合搜索关闭。

![Packed weight-gradient discarded](assets/packed-weight-gradient-discard.svg)

Experiment 219扩展显式FP32 solution registry到rank-2。三进程共同index让Qwen/DeepSeek
gate/up weight gradient快`1.077×/1.133×`，并精确命中144/168次；模型却为
`0.993×/0.996×`。index只保留为diagnostic seam，不设默认。

![FP32 weight-gradient solutions discarded](assets/fp32-weight-gradient-solutions-discard.svg)

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
| [assets/strided-batched-hipblaslt.svg](assets/strided-batched-hipblaslt.svg) | Qwen Attention batch GEMM exact-shape 加速 |
| [experiments/053-data/](experiments/053-data/) | 6 条 Event raw、错误计时和 dispatch 合同 |
| [assets/batched-attention-backward.svg](assets/batched-attention-backward.svg) | T512 两模型吞吐与 retained profile |
| [experiments/054-data/](experiments/054-data/) | 正式12条 raw、T128 fallback 与 profiler 聚合 |
| [assets/saved-attention-probabilities.svg](assets/saved-attention-probabilities.svg) | T512 吞吐、固定显存成本与 row profile |
| [experiments/055-data/](experiments/055-data/) | 正式12条 raw、fallback 与 retained profile |
| [assets/batched-attention-forward.svg](assets/batched-attention-forward.svg) | T512 两模型吞吐与 forward/全进程 Kernel 变化 |
| [experiments/056-data/](experiments/056-data/) | 正式12条 raw、T128 fallback 与 retained profiler 聚合 |
| [assets/full-batched-attention-backward.svg](assets/full-batched-attention-backward.svg) | T512 两模型吞吐与完整 batched backward 设备时间 |
| [experiments/057-data/](experiments/057-data/) | 正式12条 raw、T128 fallback 与 retained profiler 聚合 |
| [assets/block-row-causal-softmax.svg](assets/block-row-causal-softmax.svg) | T512 两模型吞吐与 softmax 前后向设备时间 |
| [experiments/058-data/](experiments/058-data/) | 正式12条 raw、T128 fallback 与 retained profiler 聚合 |
| [assets/block-column-rmsnorm-weight-gradient.svg](assets/block-column-rmsnorm-weight-gradient.svg) | T512 两模型吞吐与 RMSNorm weight-gradient 设备时间 |
| [experiments/059-data/](experiments/059-data/) | 正式12条 raw、T128 fallback 与 retained profiler 聚合 |
| [assets/inference-context-batch-matrix.svg](assets/inference-context-batch-matrix.svg) | context吞吐比、batch效率和KV Cache边界 |
| [experiments/060-data/](experiments/060-data/) | 核心108条、batch48条、long60条、无效pilot和最终schema smoke |
| [assets/batched-long-prefill-inference.svg](assets/batched-long-prefill-inference.svg) | T512/T1024 prefill自身加速、显存和profile |
| [experiments/061-data/](experiments/061-data/) | 正式24条、T128 fallback、未命中pilot和前后profile |
| [assets/full-prefill-kv-cache.svg](assets/full-prefill-kv-cache.svg) | cache prepare/end-to-end与token/full profile |
| [experiments/062-data/](experiments/062-data/) | 正式36条、T2048、两条失败修复和前后profile |
| [assets/device-rowwise-argmax.svg](assets/device-rowwise-argmax.svg) | batch shape加速、D2H字节和profile解释 |
| [experiments/063-data/](experiments/063-data/) | host/device各16条、transfer control和前后profile |
| [assets/batched-kv-cache.svg](assets/batched-kv-cache.svg) | cached batch吞吐、扩展效率、KV字节与profile |
| [experiments/064-data/](experiments/064-data/) | pilot16条、正式48条、retained B8 profile |
| [assets/bf16-kv-cache.svg](assets/bf16-kv-cache.svg) | Release吞吐、T2048 B8 Cache字节与精度门 |
| [experiments/065-data/](experiments/065-data/) | Release前后各72条、12条完整logits、profile和被拒绝向量化 |
| [assets/fused-prefix-pair-discard.svg](assets/fused-prefix-pair-discard.svg) | prepare矩阵、零D2D局部成功和长batch反例 |
| [experiments/066-data/](experiments/066-data/) | 正式72条、精度12条、profile和discard决定 |
| [assets/mixed-layer-kv-policy.svg](assets/mixed-layer-kv-policy.svg) | RMSE修复、Cache字节和显式性能代价 |
| [experiments/067-data/](experiments/067-data/) | 16组搜索、两套12-shape精度、72条formal和profile |
| [assets/targeted-prefix-pair-discard.svg](assets/targeted-prefix-pair-discard.svg) | 单FP32层D2D下降与同binary反例 |
| [experiments/068-data/](experiments/068-data/) | reference/paired各6条、精度和discard决定 |
| [assets/same-binary-kv-policy.svg](assets/same-binary-kv-policy.svg) | DeepSeek六shape策略比与跨窗口结论修正 |
| [experiments/069-data/](experiments/069-data/) | 72条交替策略raw和12条同binary summary |
| [assets/kv-policy-prompt-robustness.svg](assets/kv-policy-prompt-robustness.svg) | prompt反例、14/14 robust策略和代价 |
| [experiments/070-data/](experiments/070-data/) | layer1挑战、constant搜索、first4精度/性能数据 |
| [assets/qwen-kv-prompt-failure.svg](assets/qwen-kv-prompt-failure.svg) | Qwen pattern矩阵、context反例和FP32 fallback |
| [experiments/071-data/](experiments/071-data/) | uniform/first2挑战和T512/T2048层数搜索 |
| [assets/reference-serving-scheduler.svg](assets/reference-serving-scheduler.svg) | 请求状态机、CPU/HIP吞吐与零batch边界 |
| [experiments/072-data/](experiments/072-data/) | CPU/HIP 24条raw、8条中位数和fixed workload |
| [assets/static-batch-generation.svg](assets/static-batch-generation.svg) | HIP batch吞吐、扩展效率和static限制 |
| [experiments/073-data/](experiments/073-data/) | CPU/HIP 24条static/reference raw与8条summary |
| [assets/admission-batch-scheduler.svg](assets/admission-batch-scheduler.svg) | 分组吞吐、group数量和B4平台 |
| [experiments/074-data/](experiments/074-data/) | CPU/HIP 30条raw、10条中位数和compatibility合同 |
| [experiments/075-request-cancellation-lifecycle.md](experiments/075-request-cancellation-lifecycle.md) | 取消终态、幂等、立即Cache释放与batch排除 |
| [experiments/075-data/](experiments/075-data/) | CPU/HIP/sanitizer生命周期门摘要 |
| [assets/expanded-inference-service-matrix.svg](assets/expanded-inference-service-matrix.svg) | 长短context、batch效率、KV显存与精度分叉 |
| [experiments/076-data/](experiments/076-data/) | 120条Qwen/DeepSeek prefill、FP32/BF16 cached raw与summary |
| [assets/serving-last-logit-prefill.svg](assets/serving-last-logit-prefill.svg) | full→last吞吐、峰值、D2H和新Attention热点 |
| [experiments/077-data/](experiments/077-data/) | full/last正式矩阵、48条shape、完整logits和前后profile统计 |
| [assets/folded-gqa-discard.svg](assets/folded-gqa-discard.svg) | 性能/显存成功与完整logits反驳门 |
| [experiments/078-data/](experiments/078-data/) | 三进程候选、T2048 B8完整logits和机制profile |
| [assets/register-softmax.svg](assets/register-softmax.svg) | softmax设备时间、配对吞吐、无spill和异常复测 |
| [experiments/079-data/](experiments/079-data/) | bit-exact、A/B paired、16-shape survey、targeted recheck和profile |
| [assets/readable-fused-attention-discard.svg](assets/readable-fused-attention-discard.svg) | 无T²可读Kernel的吞吐/显存反例与backend盘点 |
| [experiments/080-data/](experiments/080-data/) | T512 B1 paired route pilot和ROCm backend inventory |
| [assets/inplace-causal-softmax.svg](assets/inplace-causal-softmax.svg) | score/prob生命周期、精确T²字节和context显存曲线 |
| [experiments/081-data/](experiments/081-data/) | bit-exact、paired memory track、16-shape survey和alias profile |
| [assets/stop-token-early-completion.svg](assets/stop-token-early-completion.svg) | 不同row终止时间、B1 Cache释放与未回收slot边界 |
| [experiments/082-data/](experiments/082-data/) | CPU/HIP GoogleTest raw、生命周期合同和环境 |
| [assets/kv-cache-clear-row.svg](assets/kv-cache-clear-row.svg) | B2完整capacity清零、其他row保护和shared-position边界 |
| [experiments/083-data/](experiments/083-data/) | CPU/HIP storage对齐、零transfer与生命周期合同 |
| [assets/kv-cache-per-row-positions.svg](assets/kv-cache-per-row-positions.svg) | uniform/divergent状态转移与严格失败边界 |
| [experiments/084-data/](experiments/084-data/) | CPU/HIP metadata、reset/advance和错误合同 |
| [assets/steady-inference-shape-memory.svg](assets/steady-inference-shape-memory.svg) | 一token一forward的Release吞吐与长batch显存 |
| [experiments/085-data/](experiments/085-data/) | semantic/Release矩阵、build-type审计和invalid runner证据 |
| [assets/deepseek-steady-profile-d2h-discard.svg](assets/deepseek-steady-profile-d2h-discard.svg) | T2048热点组成与D2H候选的B8反例 |
| [experiments/086-data/](experiments/086-data/) | rocprof聚合表、三对交替进程和allocator counter |
| [assets/immediate-default-stream-pool.svg](assets/immediate-default-stream-pool.svg) | 去除16-block相位后的allocation与吞吐门 |
| [experiments/087-data/](experiments/087-data/) | T2048/T512交替对、官方shape survey与Stream安全合同 |
| [assets/bf16x2-key-load-discard.svg](assets/bf16x2-key-load-discard.svg) | 小算子通过与百万官方logit失败的反例 |
| [experiments/088-data/](experiments/088-data/) | T2048 B1/B8完整logit误差、token和rollback门 |
| [assets/raw-packed-key-load-discard.svg](assets/raw-packed-key-load-discard.svg) | 两种pair转换得到相同失败的反驳实验 |
| [experiments/089-data/](experiments/089-data/) | public scalar恢复、相同误差和搜索关闭证据 |
| [assets/device-token-history.svg](assets/device-token-history.svg) | allocator稳定后D2H 24→3与中性性能门 |
| [experiments/090-data/](experiments/090-data/) | T2048/T512交替对、六shape survey和公共API合同 |
| [assets/normalize-cached-probabilities-discard.svg](assets/normalize-cached-probabilities-discard.svg) | 位级一致与中性负性能的对照 |
| [experiments/091-data/](experiments/091-data/) | 百万logit exact门与T2048交替性能 |
| [assets/bf16-paired-value-load-discard.svg](assets/bf16-paired-value-load-discard.svg) | 双column位级正确但lane减少的性能反例 |
| [experiments/092-data/](experiments/092-data/) | Value pair完整logit与T2048三对性能 |
| [assets/divergent-cached-row-reference.svg](assets/divergent-cached-row-reference.svg) | shared Storage上的不同position与B1 view执行 |
| [experiments/093-data/](experiments/093-data/) | 两步状态转移、CPU/HIP、dtype与serial边界 |
| [assets/slot-row-prefill.svg](assets/slot-row-prefill.svg) | 新prompt进入一个空row且旧row保持不变 |
| [experiments/094-data/](experiments/094-data/) | 单槽位prefill状态转移、CPU/HIP、dtype与失败合同 |
| [assets/serving-inference-efficiency.svg](assets/serving-inference-efficiency.svg) | N64短中长context的吞吐、显存与不稳定失败 |
| [experiments/095-data/](experiments/095-data/) | 28条paired raw、三次反驳复测和KV/forward/transfer证据 |
| [assets/continuous-slot-scheduler.svg](assets/continuous-slot-scheduler.svg) | A完成、C补位、B继续与divergent性能反例 |
| [experiments/096-data/](experiments/096-data/) | CPU/HIP状态机合同、5条divergent和3条uniform MI300X数据 |
| [assets/active-row-compaction.svg](assets/active-row-compaction.svg) | 空slot从dummy模型计算变为显式skip |
| [experiments/097-data/](experiments/097-data/) | 8条Release矩阵、12条交替A/B与inactive capacity证据 |
| [assets/positions-aware-decode.svg](assets/positions-aware-decode.svg) | 不同position通过小映射表进入同一active batch |
| [experiments/098-data/](experiments/098-data/) | 8条Release矩阵、18条交替A/B与4097 fallback证据 |
| [assets/continuous-profile-scatter-discard.svg](assets/continuous-profile-scatter-discard.svg) | 干净Kernel组成与scatter负面A/B |
| [experiments/099-data/](experiments/099-data/) | 两份pftrace、原始CSV、stdout与12条scatter交替数据 |
| [assets/packed-decode-metadata.svg](assets/packed-decode-metadata.svg) | 三份小metadata合成一个H2D Storage |
| [experiments/100-data/](experiments/100-data/) | 12条交替A/B与H2D/D2H/D2D精确counter |
| [assets/batched-slot-prefill.svg](assets/batched-slot-prefill.svg) | 相同长度prompt从8个B1变成一个[A,T]prefill |
| [experiments/101-data/](experiments/101-data/) | 18条交替A/B与logical/physical prefill计数 |
| [assets/official-continuous-serving.svg](assets/official-continuous-serving.svg) | 官方Qwen/DeepSeek吞吐、KV利用率和精度红门 |
| [experiments/102-data/](experiments/102-data/) | 24条microLLM多进程、8条PyTorch参考和逐token比较 |
| [assets/continuous-slot-sweep.svg](assets/continuous-slot-sweep.svg) | 固定8请求的S1–S8效率、KV代价和失败修复 |
| [experiments/103-data/](experiments/103-data/) | 修复前后各48进程、效率和跨slot token证据 |
| [assets/continuous-divergence.svg](assets/continuous-divergence.svg) | top-2低margin翻转和prefill-only反驳实验 |
| [experiments/104-data/](experiments/104-data/) | 18条诊断、serial counterfactual和PyTorch门 |
| [assets/prefill-row-audit.svg](assets/prefill-row-audit.svg) | B2 row交换、重复prompt和row-copy反驳 |
| [experiments/105-data/](experiments/105-data/) | 12条显式offset官方模型row/order证据 |
| [assets/prefill-layer-drift.svg](assets/prefill-layer-drift.svg) | embedding到完整logits的relative-L2增长 |
| [experiments/106-data/](experiments/106-data/) | 三对fresh B1/B2的31-stage完整值误差 |
| [assets/block0-drift.svg](assets/block0-drift.svg) | block0 Attention exact到FFN首次非零的边界 |
| [experiments/107-data/](experiments/107-data/) | 三对43-stage block0子阶段完整值误差 |
| [assets/bf16-ffn-drift.svg](assets/bf16-ffn-drift.svg) | cast exact到gate/up GEMM首次漂移 |
| [experiments/108-data/](experiments/108-data/) | 三对48-stage FFN内部完整值误差 |
| [assets/bf16-algorithm-inventory.svg](assets/bf16-algorithm-inventory.svg) | M32/M64候选集合与53个交集 |
| [experiments/109-data/](experiments/109-data/) | solution index、workspace和waves原始JSON |
| [assets/bf16-same-algorithm.svg](assets/bf16-same-algorithm.svg) | exact恢复与吞吐代价 |
| [experiments/110-data/](experiments/110-data/) | 3对精度和12条无trace性能A/B |
| [assets/qwen-common-algorithm-discard.svg](assets/qwen-common-algorithm-discard.svg) | Qwen中性性能但非exact的拒绝证据 |
| [experiments/111-data/](experiments/111-data/) | 56候选inventory、精度和性能A/B |
| [assets/qwen-algorithm-search.svg](assets/qwen-algorithm-search.svg) | 56受支持、0 exact与最佳误差 |
| [experiments/112-data/](experiments/112-data/) | 全56候选完整logits搜索 |
| [assets/request-latency.svg](assets/request-latency.svg) | short/long slot延迟权衡 |
| [experiments/113-data/](experiments/113-data/) | 48条请求级TTFT/completion/KV数据 |
| [assets/length-bucket-tradeoff.svg](assets/length-bucket-tradeoff.svg) | KV、TTFT、吞吐与完成延迟的分桶取舍 |
| [experiments/114-data/](experiments/114-data/) | 12条Release A/B、路由、token和GPU负载证据 |
| [assets/bucket-pareto-sweep.svg](assets/bucket-pareto-sweep.svg) | 1/2/4桶的Cache、吞吐和延迟Pareto曲线 |
| [experiments/115-data/](experiments/115-data/) | 18条idle-gated正式矩阵和一轮污染拒绝证据 |
| [assets/traffic-skew-tail.svg](assets/traffic-skew-tail.svg) | 固定桶的median改善与P95排队反例 |
| [experiments/116-data/](experiments/116-data/) | 36条偏斜/延迟到达矩阵和两次设备门阻断记录 |
| [assets/compatible-overflow.svg](assets/compatible-overflow.svg) | 短请求借大桶后的吞吐与P95恢复 |
| [experiments/117-data/](experiments/117-data/) | 54条三策略矩阵和一次路由合同失败 |
| [assets/slot-ratio-sweep.svg](assets/slot-ratio-sweep.svg) | short/long-heavy下静态slot最优比例翻转 |
| [experiments/118-data/](experiments/118-data/) | 48条2:6/4:4/6:2正式矩阵 |
| [assets/mi300-precision-roofline.svg](assets/mi300-precision-roofline.svg) | 128–1024的FP32/16/BF16/FP8 achieved TFLOPS |
| [experiments/119-data/](experiments/119-data/) | 20条executed precision与roofline证据 |
| [assets/large-precision-roofline.svg](assets/large-precision-roofline.svg) | 2048/4096低精度TFLOPS与峰值利用率 |
| [experiments/120-data/](experiments/120-data/) | 10条FP32-reference大GEMM证据 |
| [assets/mi300-int8-probe.svg](assets/mi300-int8-probe.svg) | 128–4096 raw INT8 executed TOPS |
| [experiments/121-data/](experiments/121-data/) | 6条INT8 exact-sample与roofline证据 |
| [assets/official-fp8-static-scale.svg](assets/official-fp8-static-scale.svg) | official FP8速度/内存与四个精度红门 |
| [experiments/122-data/](experiments/122-data/) | 36条FP32/BF16/FP8和一次worker失败 |
| [assets/fp8-global-scale-grid.svg](assets/fp8-global-scale-grid.svg) | 两个官方模型的全局scale网格最低RMS与精度门 |
| [experiments/123-data/](experiments/123-data/) | 34条fresh-process reference/scale候选和0/32过门证据 |
| [assets/fp8-scale-boundary.svg](assets/fp8-scale-boundary.svg) | activation上边界扩展后的官方模型RMS曲线 |
| [experiments/124-data/](experiments/124-data/) | 18条0.1/0.2边界扩展和0/16过门证据 |
| [assets/fp8-scale-turn.svg](assets/fp8-scale-turn.svg) | Qwen/DeepSeek全局scale误差曲线的分叉 |
| [experiments/125-data/](experiments/125-data/) | 18条0.4/0.8边界扩展和top-token反例 |
| [assets/qwen-fp8-scale-closure.svg](assets/qwen-fp8-scale-closure.svg) | Qwen扩展到3.2后的边际收益和剩余精度差距 |
| [experiments/126-data/](experiments/126-data/) | 9条Qwen-only边界收尾和诚实结论门 |
| [assets/fp8-tensor-amax-weight.svg](assets/fp8-tensor-amax-weight.svg) | per-Tensor weight scale的误差改善、剩余门差距和准备成本 |
| [experiments/127-data/](experiments/127-data/) | 36条正式矩阵、3条pilot和15条被拒绝的计时缺失数据 |
| [assets/fp8-activation-range.svg](assets/fp8-activation-range.svg) | 全层Linear输入相对固定FP8范围的冲突 |
| [experiments/128-data/](experiments/128-data/) | 208个正式activation边界、完整trace、pilot和一次合同失败 |
| [assets/fp8-device-activation-amax.svg](assets/fp8-device-activation-amax.svg) | device动态scale的RMS改善与长context性能失败 |
| [experiments/129-data/](experiments/129-data/) | 36条正式FP32/BF16/dynamic-FP8与3条pilot |
| [assets/fp8-activation-row-range.svg](assets/fp8-activation-row-range.svg) | Attention/FFN内部token row范围差异 |
| [experiments/130-data/](experiments/130-data/) | 208个Tensor的全部逐row amax与full-trace manifest |
| [assets/fp8-ffn-outer-row.svg](assets/fp8-ffn-outer-row.svg) | FFN row策略的速度恢复、精度红门与fallback次数 |
| [experiments/131-data/](experiments/131-data/) | 36条正式矩阵与3条pilot |
| [assets/fp8-device-weight-amax.svg](assets/fp8-device-weight-amax.svg) | host/device权重准备时间与冷启动加速 |
| [experiments/132-data/](experiments/132-data/) | 36条正式、fresh pilot、stale binary与fresh-build失败证据 |
| [assets/fp8-multiblock-amax.svg](assets/fp8-multiblock-amax.svg) | weight冷启动与T512 activation双重加速 |
| [experiments/133-data/](experiments/133-data/) | 两套18-worker矩阵与fresh build |
| [assets/fp8-dynamic-activation-profile.svg](assets/fp8-dynamic-activation-profile.svg) | dynamic三段与GEMM可归因时间 |
| [experiments/134-data/](experiments/134-data/) | 两模型parsed profile与kernel/API stats |
| [assets/fp8-shared-activation-quantization.svg](assets/fp8-shared-activation-quantization.svg) | QKV/gate-up共享后的T512吞吐 |
| [experiments/135-data/](experiments/135-data/) | 18条正式矩阵、verification与fresh build |
| [assets/fp8-shared-activation-profile.svg](assets/fp8-shared-activation-profile.svg) | 共享前后known-forward profile |
| [experiments/136-data/](experiments/136-data/) | 两模型复测parsed profile与stats |
| [assets/fp8-layer-drift.svg](assets/fp8-layer-drift.svg) | Qwen21/Deep27误差定位 |
| [experiments/137-data/](experiments/137-data/) | 56阶段完整差异、完整性审计与trace manifest |
| [assets/fp8-block-detail.svg](assets/fp8-block-detail.svg) | Q21/Deep27内部误差与残差相加跳变 |
| [experiments/138-data/](experiments/138-data/) | 32个内部阶段、fresh build与trace manifest |
| [assets/fp8-residual-cancellation.svg](assets/fp8-residual-cancellation.svg) | 残差误差的分子/分母精确分解 |
| [experiments/139-data/](experiments/139-data/) | 两模型完整值代数与重建门 |
| [assets/fp8-selective-block-counterfactual.svg](assets/fp8-selective-block-counterfactual.svg) | 关键层FP32的短/长context精度反例与显存代价 |
| [experiments/140-data/](experiments/140-data/) | 36个正式worker、fresh build、完整logits与策略拒绝门 |
| [assets/fp8-error-source-isolation.svg](assets/fp8-error-source-isolation.svg) | 权重/激活单侧舍入的完整logits RMS对比 |
| [experiments/141-data/](experiments/141-data/) | 24个正式worker、两种诊断合同和误差归因 |
| [assets/fp8-native-vs-roundtrip.svg](assets/fp8-native-vs-roundtrip.svg) | 原生GEMM直接扰动与最终总RMS的两条判定门 |
| [experiments/142-data/](experiments/142-data/) | 12个worker、4组直接完整向量比较和fresh build |
| [assets/fp8-output-channel-policy.svg](assets/fp8-output-channel-policy.svg) | Qwen/DeepSeek相反精度与共同T512速度回退 |
| [experiments/143-data/](experiments/143-data/) | 36个worker、scale/显存/调用计数和keep门 |
| [assets/fp8-output-column-native-probe.svg](assets/fp8-output-column-native-probe.svg) | outer-vector拒绝与scalar+post有效路径 |
| [experiments/144-data/](experiments/144-data/) | fresh GTest JSON、6个模型worker和能力门 |
| [assets/fp8-weight-reconstruction-audit.svg](assets/fp8-weight-reconstruction-audit.svg) | Attention/FFN/head权重重建与模型放大反例 |
| [experiments/145-data/](experiments/145-data/) | 365个真实Linear、分组SSE和外部诊断边界 |
| [assets/fp8-output-head-only.svg](assets/fp8-output-head-only.svg) | 同revision零数值变化、微小速度/显存代价 |
| [experiments/146-data/](experiments/146-data/) | 候选/control共72 worker与错误基线审计 |
| [assets/fp8-attention-only.svg](assets/fp8-attention-only.svg) | 7/8误差改善、Qwen长RMS反例和T512速度门 |
| [experiments/147-data/](experiments/147-data/) | 同revision候选/control、72 worker和scope计数 |
| [assets/fp8-attention-output-only.svg](assets/fp8-attention-output-only.svg) | Qwen零回归、Deep改善与T512 keep门 |
| [experiments/148-data/](experiments/148-data/) | O-only/control共72 worker和targeted keep证据 |
| [assets/fp8-clipped-pilot-invalid.svg](assets/fp8-clipped-pilot-invalid.svg) | 外部GPU争用时间线和严格拒绝门 |
| [experiments/149-data/](experiments/149-data/) | 0/4有效fraction、污染行排除和重试合同 |
| [assets/fp8-fraction-pilot-workload-invalid.svg](assets/fp8-fraction-pilot-workload-invalid.svg) | retained/执行weight起点不一致 |
| [experiments/150-data/](experiments/150-data/) | 20 worker执行合同与4/4 baseline mismatch |
| [assets/fp8-clipped-coarse-grid.svg](assets/fp8-clipped-coarse-grid.svg) | fraction下降时worst RMS/Max急剧恶化 |
| [experiments/151-data/](experiments/151-data/) | 有效20-worker coarse grid与精细网格交接 |
| [assets/fp8-clipped-fine-grid.svg](assets/fp8-clipped-fine-grid.svg) | 0.95/0.9/0.85的worst RMS恶化曲线 |
| [experiments/152-data/](experiments/152-data/) | 精细20-worker网格与模型clipping关闭门 |
| [assets/fp8-e5-activation-discard.svg](assets/fp8-e5-activation-discard.svg) | E5相对E4的八项完整logits误差回归 |
| [experiments/153-data/](experiments/153-data/) | E5/control共72 worker、格式与调度计数及拒绝门 |
| [assets/fp8-layer-leave-one-out.svg](assets/fp8-layer-leave-one-out.svg) | Qwen/DeepSeek全部单层FP32敏感度与反例 |
| [experiments/154-data/](experiments/154-data/) | 56行完整logits、52层排名与routing合同 |
| [assets/fp8-qwen-layer9-formal-discard.svg](assets/fp8-qwen-layer9-formal-discard.svg) | Qwen layer9短上下文改善与长上下文反转 |
| [experiments/155-data/](experiments/155-data/) | candidate/control共36 worker、显存/速度/精度拒绝门 |
| [assets/block-reduction-determinism.svg](assets/block-reduction-determinism.svg) | reduction数据竞争从20/20不同到bit-exact |
| [experiments/156-data/](experiments/156-data/) | 旧revision反例、20进程门与三进程性能证据 |
| [assets/adamw-correctness-before-timing.svg](assets/adamw-correctness-before-timing.svg) | 完整optimizer状态门、真实参数量速度与Scalar保留结论 |
| [experiments/157-adamw-correctness-before-timing.md](experiments/157-adamw-correctness-before-timing.md) | exact key/cache、15进程MI300矩阵与端到端中性回归 |
| [assets/cooperative-bias-gradient.svg](assets/cooperative-bias-gradient.svg) | 32-row边界、真实宽度算子加速与两模型整机收益 |
| [experiments/158-cooperative-bias-gradient.md](experiments/158-cooperative-bias-gradient.md) | 78行完整输出、同revision A/B与rocprofv3归因 |
| [assets/post-bias-training-profile.svg](assets/post-bias-training-profile.svg) | 每训练step分类、load-only反例与下一热点选择 |
| [experiments/159-post-bias-training-profile.md](experiments/159-post-bias-training-profile.md) | 1步/3步相位差分与53.47% GEMM结论 |
| [assets/bf16-training-solution-discard.svg](assets/bf16-training-solution-discard.svg) | 八shape算子收益、两种模型策略与拒绝门 |
| [experiments/160-bf16-training-solution-discard.md](experiments/160-bf16-training-solution-discard.md) | 1536候选、24进程和同revision整机反驳 |
| [assets/tied-embedding-sparse-add.svg](assets/tied-embedding-sparse-add.svg) | gradient来源、Qwen峰值与稀疏累加路径 |
| [experiments/161-tied-embedding-sparse-add.md](experiments/161-tied-embedding-sparse-add.md) | 71.2%元素归因、两模型A/B和profile归因 |
| [assets/attention-rope-layout-fusion.svg](assets/attention-rope-layout-fusion.svg) | Q/K布局复制、两模型T512与rocprofv3归因 |
| [experiments/162-attention-rope-layout-fusion.md](experiments/162-attention-rope-layout-fusion.md) | 前向/双梯度门、60% copy消除与保留结论 |
| [assets/attention-interleaved-pv.svg](assets/attention-interleaved-pv.svg) | 交错head地址、五shape算子速度与边界反例 |
| [experiments/163-attention-interleaved-pv.md](experiments/163-attention-interleaved-pv.md) | 30进程完整输出与hipBLASLt布局能力证据 |
| [assets/attention-context-layout-fusion.svg](assets/attention-context-layout-fusion.svg) | BTHD前后向、零strided-copy与两模型整机门 |
| [experiments/164-attention-context-layout-fusion.md](experiments/164-attention-context-layout-fusion.md) | output/dP/dV/QKV梯度、T512 A/B与profile |
| [assets/post-layout-training-profile.svg](assets/post-layout-training-profile.svg) | 零copy后的Kernel重排与interleaved plan假设 |
| [experiments/165-post-layout-training-profile.md](experiments/165-post-layout-training-profile.md) | 1步/3步相位差分、关闭路线与下一节点 |
| [assets/attention-layout-plan-cache-discard.svg](assets/attention-layout-plan-cache-discard.svg) | 算子收益与整机拒绝的并列证据 |
| [experiments/166-attention-layout-plan-cache-discard.md](experiments/166-attention-layout-plan-cache-discard.md) | exact cache路由、24算子进程与12模型进程 |
| [assets/attention-gemm-scale-fusion-discard.svg](assets/attention-gemm-scale-fusion-discard.svg) | scale Kernel归零与两模型混合拒绝门 |
| [experiments/167-attention-gemm-scale-fusion-discard.md](experiments/167-attention-gemm-scale-fusion-discard.md) | alpha算子、12模型进程、rounding与profile |
| [assets/paired-gqa-repeat-discard.svg](assets/paired-gqa-repeat-discard.svg) | repeat Kernel减半与Qwen反例 |
| [experiments/168-paired-gqa-repeat-discard.md](experiments/168-paired-gqa-repeat-discard.md) | 成对前后向、12模型进程与完整profile |
| [assets/gqa-zero-stride-value-broadcast.svg](assets/gqa-zero-stride-value-broadcast.svg) | Qwen/DeepSeek相反的零stride P×V结果 |
| [experiments/169-gqa-zero-stride-value-broadcast.md](experiments/169-gqa-zero-stride-value-broadcast.md) | 30算子进程、MHA反例与width-selective下一门 |
| [assets/selective-gqa-value-broadcast-discard.svg](assets/selective-gqa-value-broadcast-discard.svg) | width选择、Deep整机拒绝与profile抵消 |
| [experiments/170-selective-gqa-value-broadcast-discard.md](experiments/170-selective-gqa-value-broadcast-discard.md) | dP/QKV梯度、12模型进程与Deep profile |
| [assets/forward-only-gqa-value-broadcast-discard.svg](assets/forward-only-gqa-value-broadcast-discard.svg) | forward-only整机/参数/profile最终拒绝 |
| [experiments/171-forward-only-gqa-value-broadcast-discard.md](experiments/171-forward-only-gqa-value-broadcast-discard.md) | zero-stride模型路线关闭证据 |
| [assets/unique-gradient-inplace-add-discard.svg](assets/unique-gradient-inplace-add-discard.svg) | 真正少allocation但不减少device工作的反例 |
| [experiments/172-unique-gradient-inplace-add-discard.md](experiments/172-unique-gradient-inplace-add-discard.md) | 独占owner合同、两模型A/B与rocprofv3拒绝门 |
| [assets/hip-graph-submission-crossover.svg](assets/hip-graph-submission-crossover.svg) | 1/8节点反例与32–512节点Graph收益拐点 |
| [experiments/173-hip-graph-runtime.md](experiments/173-hip-graph-runtime.md) | caller-owned capture、sticky-error恢复与模型阻塞边界 |
| [assets/hip-graph-gemm-discard.svg](assets/hip-graph-gemm-discard.svg) | Qwen边缘收益与DeepSeek vendor-GEMM反例 |
| [experiments/174-hip-graph-gemm-discard.md](experiments/174-hip-graph-gemm-discard.md) | stable matmul输出、36进程矩阵与profile拒绝门 |
| [assets/scoped-model-stream-discard.svg](assets/scoped-model-stream-discard.svg) | Stream路由成功但Storage lifetime导致完整logits错误 |
| [experiments/175-scoped-model-stream-discard.md](experiments/175-scoped-model-stream-discard.md) | 三次稳定失败、候选移除与deferred-release前置条件 |
| [assets/deferred-hip-deallocation.svg](assets/deferred-hip-deallocation.svg) | 310次同步消除、速度与pending-memory代价 |
| [experiments/176-deferred-hip-deallocation.md](experiments/176-deferred-hip-deallocation.md) | explicit lifetime合同、36进程矩阵与overflow测试 |
| [assets/scoped-deferred-model-stream.svg](assets/scoped-deferred-model-stream.svg) | 8个官方workload速度比与14.5GiB代价 |
| [experiments/177-scoped-deferred-model-stream.md](experiments/177-scoped-deferred-model-stream.md) | bit-exact模型Stream、48进程矩阵与allocator归因 |
| [assets/per-device-hipblaslt-handles.svg](assets/per-device-hipblaslt-handles.svg) | RCCL 6/11→11/11与四项单卡非回归 |
| [experiments/178-per-device-hipblaslt-handles.md](experiments/178-per-device-hipblaslt-handles.md) | handle设备所有权、交替GPU测试与12进程矩阵 |
| [assets/stream-ordered-allocator.svg](assets/stream-ordered-allocator.svg) | eager async与Graph allocation-node速度/地址反例 |
| [experiments/179-stream-ordered-allocator.md](experiments/179-stream-ordered-allocator.md) | 72进程矩阵、pool high-water和profile关闭门 |
| [assets/activation-arena.svg](assets/activation-arena.svg) | stable two-slot eager/Graph速度与setup回本次数 |
| [experiments/180-activation-arena.md](experiments/180-activation-arena.md) | 72进程矩阵、compute-only Graph和liveness合同 |
| [assets/arena-ffn.svg](assets/arena-ffn.svg) | Qwen/DeepSeek四算子FFN速度与短行反例 |
| [experiments/181-arena-ffn.md](experiments/181-arena-ffn.md) | official shape、36进程、四节点Graph与profile |
| [assets/bf16-arena-ffn.svg](assets/bf16-arena-ffn.svg) | BF16 caller-owned FFN六shape速度与Graph反例 |
| [experiments/182-bf16-arena-ffn.md](experiments/182-bf16-arena-ffn.md) | 54进程、direct/fallback节点与分配profile |
| [assets/bf16-ffn-arena-model.svg](assets/bf16-ffn-arena-model.svg) | 两模型五case完整推理比率与选择边界 |
| [experiments/183-bf16-ffn-arena-model.md](experiments/183-bf16-ffn-arena-model.md) | 60进程完整logits、分配profile与全局策略拒绝 |
| [assets/bf16-ffn-arena-selective.svg](assets/bf16-ffn-arena-selective.svg) | rows≥512 eligible与八个精确bypass case |
| [experiments/184-bf16-ffn-arena-selective.md](experiments/184-bf16-ffn-arena-selective.md) | 两模型长prefill keep、60进程与profile |
| [assets/bf16-qkv-arena-discard.svg](assets/bf16-qkv-arena-discard.svg) | QKV eligible/bypass完整模型失败图 |
| [experiments/185-bf16-qkv-arena-discard.md](experiments/185-bf16-qkv-arena-discard.md) | 分配下降但T512仅1.004×/1.005× |
| [assets/allocation-source-attribution.svg](assets/allocation-source-attribution.svg) | Qwen/DeepSeek T512 source bytes堆叠图 |
| [experiments/186-allocation-source-attribution.md](experiments/186-allocation-source-attribution.md) | 6进程确定性分布与Attention core选择 |
| [assets/attention-core-arena-discard.svg](assets/attention-core-arena-discard.svg) | 最大allocation source的整模失败证据 |
| [experiments/187-attention-core-arena-discard.md](experiments/187-attention-core-arena-discard.md) | 600/700分配下降但仅1.004×/1.002× |
| [assets/fp32-attention-solutions.svg](assets/fp32-attention-solutions.svg) | 四个FP32 QK/PV exact solution加速 |
| [experiments/188-fp32-attention-solutions.md](experiments/188-fp32-attention-solutions.md) | 12进程、64共同候选与完整输出门 |
| [assets/fp32-attention-model-gate.svg](assets/fp32-attention-model-gate.svg) | QK/PV/both整模速度与bit-exact门 |
| [experiments/189-fp32-attention-model-gate.md](experiments/189-fp32-attention-model-gate.md) | 精确注册、24进程与默认策略拒绝 |
| [assets/bf16-grouped-qkv.svg](assets/bf16-grouped-qkv.svg) | pointer-stable算子收益与两模型整模分叉 |
| [experiments/190-bf16-grouped-qkv.md](experiments/190-bf16-grouped-qkv.md) | phase delta、24进程、plan cache与默认拒绝 |
| [assets/bf16-grouped-qkv-expanded.svg](assets/bf16-grouped-qkv-expanded.svg) | 64候选、两模型steady keep与setup gate |
| [experiments/191-bf16-grouped-qkv-expanded.md](experiments/191-bf16-grouped-qkv-expanded.md) | user arguments、24进程与显式预热策略 |
| [assets/bf16-grouped-qkv-prewarm.svg](assets/bf16-grouped-qkv-prewarm.svg) | lazy与prewarm首请求时间线 |
| [experiments/192-bf16-grouped-qkv-prewarm.md](experiments/192-bf16-grouped-qkv-prewarm.md) | 18进程、prewarm API与admission边界 |
| [assets/hipblaslt-preload.svg](assets/hipblaslt-preload.svg) | 全kernel预载的两模型冷启动反例 |
| [experiments/193-hipblaslt-preload.md](experiments/193-hipblaslt-preload.md) | 18进程、wall/forward/显存与策略拒绝 |
| [assets/bf16-exact-startup.svg](assets/bf16-exact-startup.svg) | exact gate/up算子、cold和steady三层对照 |
| [experiments/194-bf16-exact-startup.md](experiments/194-bf16-exact-startup.md) | 6 tuner + 24模型进程与bit-exact拒绝 |
| [assets/bf16-grouped-gate-up.svg](assets/bf16-grouped-gate-up.svg) | 双gate/up稳定、device arguments与重初始化对照 |
| [experiments/195-bf16-grouped-gate-up.md](experiments/195-bf16-grouped-gate-up.md) | 6进程、64候选与FFN Arena接入门 |
| [assets/bf16-grouped-gate-up-model.svg](assets/bf16-grouped-gate-up-model.svg) | 两模型吞吐与每层少一次GEMM提交 |
| [experiments/196-bf16-grouped-gate-up-model.md](experiments/196-bf16-grouped-gate-up-model.md) | exact registry、12进程、setup/peak/profile gate |
| [assets/bf16-grouped-composition.svg](assets/bf16-grouped-composition.svg) | baseline、QKV、gate/up、both四策略吞吐 |
| [experiments/197-bf16-grouped-composition.md](experiments/197-bf16-grouped-composition.md) | 24进程、双registry dispatch与组合setup |
| [assets/bf16-grouped-shape-matrix.svg](assets/bf16-grouped-shape-matrix.svg) | rows256/1024两模型两projection user-args收益 |
| [experiments/198-bf16-grouped-shape-matrix.md](experiments/198-bf16-grouped-shape-matrix.md) | 24进程、winner集合与重初始化反驳 |
| [assets/bf16-grouped-shape-models.svg](assets/bf16-grouped-shape-models.svg) | B1/T256、B1/T1024、B2/T512完整模型 |
| [experiments/199-bf16-grouped-shape-models.md](experiments/199-bf16-grouped-shape-models.md) | 36进程、batch-row top-1与CLI导出修复 |
| [assets/bf16-grouped-composed-profile.svg](assets/bf16-grouped-composed-profile.svg) | 组合后GEMM calls与剩余时间占比 |
| [experiments/200-bf16-grouped-composed-profile.md](experiments/200-bf16-grouped-composed-profile.md) | 四trace、phase delta与下一热点选择 |
| [assets/hf-strided-copy-sources.svg](assets/hf-strided-copy-sources.svg) | Attention layout/core剩余copy字节 |
| [experiments/201-hf-strided-copy-sources.md](experiments/201-hf-strided-copy-sources.md) | source-aware diagnostics、6进程与BTHD选择 |
| [assets/inference-bthd-attention.svg](assets/inference-bthd-attention.svg) | copy 96/112→0与完整模型速度 |
| [experiments/202-inference-bthd-attention.md](experiments/202-inference-bthd-attention.md) | 24进程、bit-exact、peak与显式fallback |
| [assets/inference-bthd-shape-models.svg](assets/inference-bthd-shape-models.svg) | BTHD长短序列与batch速度 |
| [experiments/203-inference-bthd-shape-models.md](experiments/203-inference-bthd-shape-models.md) | 42进程、Attention copy 0与B2 residual |
| [assets/inference-bthd-profile.svg](assets/inference-bthd-profile.svg) | BTHD前后Kernel时间与新热点 |
| [experiments/204-inference-bthd-profile.md](experiments/204-inference-bthd-profile.md) | 四trace、strided归零与cast候选 |
| [assets/inference-bthd-bf16-qk.svg](assets/inference-bthd-bf16-qk.svg) | 五进程整模收益与cast删除证据 |
| [experiments/205-inference-bthd-bf16-qk.md](experiments/205-inference-bthd-bf16-qk.md) | BF16 Q/K直入融合RoPE；三进程反例与五进程正式门 |
| [assets/inference-bthd-bf16-qk-shapes.svg](assets/inference-bthd-bf16-qk-shapes.svg) | 两模型三case五进程收益与1.01门 |
| [experiments/206-inference-bthd-bf16-qk-shapes.md](experiments/206-inference-bthd-bf16-qk-shapes.md) | B1/T256、B1/T1024、B2/T512完整矩阵 |
| [assets/causal-softmax-128-discard.svg](assets/causal-softmax-128-discard.svg) | 128线程六shape算子矩阵与拒绝门 |
| [experiments/207-causal-softmax-128-discard.md](experiments/207-causal-softmax-128-discard.md) | DeepSeek T512反例；模型/CLI策略未进入 |
| [assets/bf16-repeat-fusion-discard.svg](assets/bf16-repeat-fusion-discard.svg) | BF16 V cast+repeat八shape矩阵 |
| [experiments/208-bf16-repeat-fusion-discard.md](experiments/208-bf16-repeat-fusion-discard.md) | 小B1加速与B2反例；模型接入取消 |
| [assets/post-bf16-qk-saturation.svg](assets/post-bf16-qk-saturation.svg) | 当前Kernel占比、理论上限与反驳链 |
| [experiments/209-post-bf16-qk-saturation.md](experiments/209-post-bf16-qk-saturation.md) | 推理微融合局部饱和；online Attention合同 |
| [assets/training-add-rms-norm-discard.svg](assets/training-add-rms-norm-discard.svg) | 训练融合少72次launch但端到端回退 |
| [experiments/210-training-add-rms-norm-fusion-discard.md](experiments/210-training-add-rms-norm-fusion-discard.md) | 分支梯度完整对齐与两模型拒绝门 |
| [assets/multi-tensor-adamw-discard.svg](assets/multi-tensor-adamw-discard.svg) | 290/339次AdamW启动合一与DeepSeek带宽反例 |
| [experiments/211-multi-tensor-adamw-discard.md](experiments/211-multi-tensor-adamw-discard.md) | pinned异步metadata、五进程模型门与隔离profile |
| [assets/training-bf16-shared-activation-discard.svg](assets/training-bf16-shared-activation-discard.svg) | 共享cast的三策略模型反例与结构命中 |
| [experiments/212-training-bf16-shared-activation-discard.md](experiments/212-training-bf16-shared-activation-discard.md) | 56进程、双模型profile与多输出梯度门 |
| [assets/post-training-micro-saturation.svg](assets/post-training-micro-saturation.svg) | 去除加载后的训练Kernel占比与完美消除上限 |
| [experiments/213-post-training-micro-saturation.md](experiments/213-post-training-micro-saturation.md) | GEMM+AdamW占72.71%/83.77%的收口证据 |
| [assets/bf16-adamw-moments.svg](assets/bf16-adamw-moments.svg) | 两模型吞吐、optimizer与峰值显存对照 |
| [experiments/214-bf16-adamw-moments-partial.md](experiments/214-bf16-adamw-moments-partial.md) | BF16 moment、计时修复、multi-tensor反例与partial keep |
| [assets/hybrid-bf16-adamw.svg](assets/hybrid-bf16-adamw.svg) | 六阈值搜索、1M正式门和16M反例 |
| [experiments/215-hybrid-bf16-adamw.md](experiments/215-hybrid-bf16-adamw.md) | 小Tensor分层合并与HIP Auto边界 |
| [assets/post-hybrid-training-profile.svg](assets/post-hybrid-training-profile.svg) | Hybrid前后总Kernel/AdamW与新类别占比 |
| [experiments/216-post-hybrid-training-profile.md](experiments/216-post-hybrid-training-profile.md) | 差分profile与训练GEMM下一合同 |
| [assets/grouped-weight-gradient-discard.svg](assets/grouped-weight-gradient-discard.svg) | direct/materialized八格能力失败 |
| [experiments/217-grouped-weight-gradient-discard.md](experiments/217-grouped-weight-gradient-discard.md) | FP32 GroupedGemm能力门与路由拒绝 |
| [assets/packed-weight-gradient-discard.svg](assets/packed-weight-gradient-discard.svg) | 四项pack+大GEMM速度与显存代价 |
| [experiments/218-packed-weight-gradient-discard.md](experiments/218-packed-weight-gradient-discard.md) | 12进程完整输出反例与组合搜索关闭 |
| [assets/fp32-weight-gradient-solutions-discard.svg](assets/fp32-weight-gradient-solutions-discard.svg) | operator winner、exact hits与模型反例 |
| [experiments/219-fp32-weight-gradient-solutions-discard.md](experiments/219-fp32-weight-gradient-solutions-discard.md) | rank-2 registry和solution默认拒绝 |
| [assets/training-graph-capture-boundary.svg](assets/training-graph-capture-boundary.svg) | FP32/BF16四阶段capture与两类阻塞条件 |
| [experiments/220-training-graph-capture-boundary.md](experiments/220-training-graph-capture-boundary.md) | 24进程、失败恢复与完整训练Graph拒绝 |
| [assets/adamw-graph-replay.svg](assets/adamw-graph-replay.svg) | device step下FP32/BF16大小/Tensor数性能边界 |
| [experiments/221-adamw-device-step-graph.md](experiments/221-adamw-device-step-graph.md) | 60进程、53步状态对齐与显式partial keep |
| [assets/adamw-graph-multi.svg](assets/adamw-graph-multi.svg) | per-Tensor与两节点multi Graph的对数速度矩阵 |
| [experiments/222-adamw-stable-descriptor-multi-graph.md](experiments/222-adamw-stable-descriptor-multi-graph.md) | 90进程、immutable descriptor与大FP32反例 |
| [assets/gradient-address-stability.svg](assets/gradient-address-stability.svg) | 六case稳定/变化gradient字节与Tensor数 |
| [experiments/223-gradient-address-stability.md](experiments/223-gradient-address-stability.md) | 18进程、context反例与Graph eligibility边界 |
| [assets/optimizer-graph-model-preflight.svg](assets/optimizer-graph-model-preflight.svg) | Stream创建到安全拒绝的完整因果链 |
| [experiments/224-optimizer-graph-model-preflight.md](experiments/224-optimizer-graph-model-preflight.md) | 12进程、pool disable与零launch模型反例 |
| [assets/quiescent-allocator-handoff.svg](assets/quiescent-allocator-handoff.svg) | default/Graph阶段状态机与四case救回结果 |
| [experiments/225-quiescent-allocator-handoff.md](experiments/225-quiescent-allocator-handoff.md) | 24进程、三次handoff与Deep长上下文反例 |
| [assets/optimizer-graph-model-gate.svg](assets/optimizer-graph-model-gate.svg) | optimizer/完整step双指标模型反例 |
| [experiments/226-optimizer-graph-model-gate.md](experiments/226-optimizer-graph-model-gate.md) | 21进程精确状态、零metadata copy与模型拒绝 |
| [assets/rocwmma-qk-tile.svg](assets/rocwmma-qk-tile.svg) | T16–2048、D64/128的rocWMMA/标量/hipBLASLt边界 |
| [experiments/227-rocwmma-qk-tile.md](experiments/227-rocwmma-qk-tile.md) | 矩阵单元能力门、完整输出与online Attention原型准入 |
| [assets/rocwmma-online-attention.svg](assets/rocwmma-online-attention.svg) | 两种真实GQA配置的candidate/current曲线与T2048显存删除量 |
| [experiments/228-rocwmma-online-attention.md](experiments/228-rocwmma-online-attention.md) | online max/sum、MFMA QK/PV、失败修正与operator准入 |
| [assets/rocwmma-online-operator.svg](assets/rocwmma-online-operator.svg) | 10个native与4个fallback公共API速度/路由边界 |
| [experiments/229-rocwmma-online-operator.md](experiments/229-rocwmma-online-operator.md) | 公共API、CMake feature、PyTorch与模型门准入 |
| [assets/rocwmma-online-model-discard.svg](assets/rocwmma-online-model-discard.svg) | 六格整模吞吐、显存节省与logit失败 |
| [experiments/230-rocwmma-online-model-discard.md](experiments/230-rocwmma-online-model-discard.md) | 36进程整模反例与模型路由拒绝 |
| [assets/rocwmma-direct-bf16-model-discard.svg](assets/rocwmma-direct-bf16-model-discard.svg) | 三cast与direct-BF16整模速度配对 |
| [experiments/231-rocwmma-direct-bf16-model-discard.md](experiments/231-rocwmma-direct-bf16-model-discard.md) | direct-BF16反驳与online模型track关闭 |
| [assets/current-inference-profile.svg](assets/current-inference-profile.svg) | 当前B1T1024默认路径分类占比 |
| [experiments/232-current-inference-profile.md](experiments/232-current-inference-profile.md) | 4进程rocprof、phase delta与新GEMM合同 |
| [assets/fp32-attention-t1024-discard.svg](assets/fp32-attention-t1024-discard.svg) | 四个operator winner与两个整模反例 |
| [experiments/233-fp32-attention-t1024-discard.md](experiments/233-fp32-attention-t1024-discard.md) | descriptor mismatch、logits与默认拒绝 |
| [assets/bf16-swiglu-vector-discard.svg](assets/bf16-swiglu-vector-discard.svg) | operator加速与整模收益落差 |
| [experiments/234-bf16-swiglu-vector-discard.md](experiments/234-bf16-swiglu-vector-discard.md) | 预分配测量、bit-exact门与Auto拒绝 |
| [assets/bf16-grouped-swish-discard.svg](assets/bf16-grouped-swish-discard.svg) | grouped epilogue局部收益和整模反例 |
| [experiments/235-bf16-grouped-swish-discard.md](experiments/235-bf16-grouped-swish-discard.md) | 64-candidate能力门、same-binary A/B与路线关闭 |
| [assets/bf16-rms-norm-output.svg](assets/bf16-rms-norm-output.svg) | Qwen/DeepSeek Event与wall加速 |
| [experiments/236-bf16-rms-norm-output.md](experiments/236-bf16-rms-norm-output.md) | GPU reference修正、位级门和operator准入 |
| [assets/bf16-ffn-norm-model.svg](assets/bf16-ffn-norm-model.svg) | 整模吞吐、allocation与精度门 |
| [experiments/237-bf16-ffn-norm-model.md](experiments/237-bf16-ffn-norm-model.md) | Arena直入、fallback修正和默认决定 |
| [assets/post-bf16-ffn-norm-profile.svg](assets/post-bf16-ffn-norm-profile.svg) | 融合前后Kernel时间与cast调用 |
| [experiments/238-post-bf16-ffn-norm-profile.md](experiments/238-post-bf16-ffn-norm-profile.md) | 四进程phase delta与下一合同 |
| [assets/bf16-attention-norm-model.svg](assets/bf16-attention-norm-model.svg) | 整模吞吐、峰值、allocation和精度 |
| [experiments/239-bf16-attention-norm-model.md](experiments/239-bf16-attention-norm-model.md) | QKV precast合同与默认路由 |
| [assets/post-bf16-attention-norm-profile.svg](assets/post-bf16-attention-norm-profile.svg) | 两Norm融合前后Kernel和cast |
| [experiments/240-post-bf16-attention-norm-profile.md](experiments/240-post-bf16-attention-norm-profile.md) | 剩余一进一出边界和下一合同 |
| [assets/bf16-pv-output-discard.svg](assets/bf16-pv-output-discard.svg) | BTHD/GQA两种descriptor能力拒绝 |
| [experiments/241-bf16-pv-output-discard.md](experiments/241-bf16-pv-output-discard.md) | status 6、0计时与API撤回 |
| [assets/bf16-value-pv-discard.svg](assets/bf16-value-pv-discard.svg) | BF16 V的BTHD/GQA能力拒绝 |
| [experiments/242-bf16-value-pv-discard.md](experiments/242-bf16-value-pv-discard.md) | 反向cast能力门与vendor路线关闭 |
| [assets/inference-local-saturation.svg](assets/inference-local-saturation.svg) | 剩余cast上限与六条关闭路线 |
| [experiments/243-inference-local-saturation.md](experiments/243-inference-local-saturation.md) | 当前推理局部策略搜索的停止门 |
| [assets/current-training-profile.svg](assets/current-training-profile.svg) | 当前B1T512训练Kernel时间分布 |
| [experiments/244-current-training-profile.md](experiments/244-current-training-profile.md) | 四进程重采样与下一训练架构合同 |
| [assets/bf16-weight-gradient-shapes.svg](assets/bf16-weight-gradient-shapes.svg) | 六个真实weight-gradient shape的胜负矩阵 |
| [experiments/245-bf16-weight-gradient-shapes.md](experiments/245-bf16-weight-gradient-shapes.md) | BF16数学门、四个反例与gate/up准入 |
| [assets/bf16-weight-gradient-model.svg](assets/bf16-weight-gradient-model.svg) | 两模型吞吐、路由、loss与峰值门 |
| [experiments/246-bf16-weight-gradient-model.md](experiments/246-bf16-weight-gradient-model.md) | 接线反例、短模型门与显式保留决定 |
| [assets/bf16-weight-gradient-trajectory-discard.svg](assets/bf16-weight-gradient-trajectory-discard.svg) | 五个长轨迹门与模型路由拒绝 |
| [experiments/247-bf16-weight-gradient-trajectory-discard.md](experiments/247-bf16-weight-gradient-trajectory-discard.md) | 20-step、完整参数和候选清理 |
| [assets/bf16-weight-gradient-allocation-attribution.svg](assets/bf16-weight-gradient-allocation-attribution.svg) | route、逻辑分配与cast字节恒等式 |
| [experiments/248-bf16-weight-gradient-allocation-attribution.md](experiments/248-bf16-weight-gradient-allocation-attribution.md) | cache复用、零backend增量与workspace前置门 |
| [assets/bf16-weight-gradient-workspace-discard.svg](assets/bf16-weight-gradient-workspace-discard.svg) | allocating/preallocated Event与wall反例 |
| [experiments/249-bf16-weight-gradient-workspace-discard.md](experiments/249-bf16-weight-gradient-workspace-discard.md) | 0/2 workspace门与API拒绝 |
| [assets/training-local-saturation.svg](assets/training-local-saturation.svg) | 当前profile、cast上限与六条关闭路线 |
| [experiments/250-training-local-saturation.md](experiments/250-training-local-saturation.md) | 训练局部策略停止门与下一架构尺度 |
| [assets/current-data-parallel-audit.svg](assets/current-data-parallel-audit.svg) | 当前双卡阶段占比与production gap |
| [experiments/251-current-data-parallel-audit.md](experiments/251-current-data-parallel-audit.md) | RCCL14/14、20-step与第一production合同 |
| [assets/data-parallel-verification-interval.svg](assets/data-parallel-verification-interval.svg) | 三种审计policy的steady total与检查次数 |
| [experiments/252-data-parallel-verification-interval.md](experiments/252-data-parallel-verification-interval.md) | 隐式同步修复、loss等价与默认兼容 |
| [assets/data-parallel-bucket-matrix.svg](assets/data-parallel-bucket-matrix.svg) | bucket count、通信与total矩阵 |
| [experiments/253-data-parallel-bucket-matrix.md](experiments/253-data-parallel-bucket-matrix.md) | tiny overlap workload拒绝与Model-S交接 |
| [assets/data-parallel-model-s-buckets.svg](assets/data-parallel-model-s-buckets.svg) | Model-S bucket/communication/total/peak矩阵 |
| [experiments/254-data-parallel-model-s-buckets.md](experiments/254-data-parallel-model-s-buckets.md) | 3-bucket reducer baseline与copy归因交接 |
| [assets/data-parallel-bucket-copy-attribution.svg](assets/data-parallel-bucket-copy-attribution.svg) | 126 Tensor、228 copy与374MB恒等式 |
| [experiments/255-data-parallel-bucket-copy-attribution.md](experiments/255-data-parallel-bucket-copy-attribution.md) | backend分配热点与persistent reducer准入 |
| [assets/data-parallel-inplace-average.svg](assets/data-parallel-inplace-average.svg) | allocating/in-place通信与total A/B |
| [experiments/256-data-parallel-inplace-average.md](experiments/256-data-parallel-inplace-average.md) | 地址稳定、RCCL门与默认保留 |
| [assets/data-parallel-persistent-buckets.svg](assets/data-parallel-persistent-buckets.svg) | persistent通信/total收益与live/peak代价 |
| [experiments/257-data-parallel-persistent-buckets.md](experiments/257-data-parallel-persistent-buckets.md) | 后续allocation归零、保持显式与bucket-view交接 |
| [assets/data-parallel-gradient-bucket-views.svg](assets/data-parallel-gradient-bucket-views.svg) | 三策略速度、live/peak与unpack删除对比 |
| [experiments/258-data-parallel-gradient-bucket-views.md](experiments/258-data-parallel-gradient-bucket-views.md) | 114个Storage/copy删除与direct-backward交接 |
| [assets/data-parallel-direct-bucket-gradient-discard.svg](assets/data-parallel-direct-bucket-gradient-discard.svg) | copy归零但forward/backward回退的反例 |
| [experiments/259-data-parallel-direct-bucket-gradients-discard.md](experiments/259-data-parallel-direct-bucket-gradients-discard.md) | 模型route拒绝与producer out-kernel交接 |
| [assets/gradient-producer-out-matrix.svg](assets/gradient-producer-out-matrix.svg) | 五shape Event/Wall与allocation 1→0算子门 |
| [experiments/260-gradient-producer-out-matrix.md](experiments/260-gradient-producer-out-matrix.md) | CPU/HIP/PyTorch对齐与scoped Autograd准入 |
| [assets/scoped-autograd-gradient-producer-discard.svg](assets/scoped-autograd-gradient-producer-discard.svg) | allocation减少但Autograd无收益的反例 |
| [experiments/261-scoped-autograd-gradient-producer-discard.md](experiments/261-scoped-autograd-gradient-producer-discard.md) | 0/5门、route撤回与gradient-ready交接 |
| [assets/data-parallel-gradient-ready-order.svg](assets/data-parallel-gradient-ready-order.svg) | 57参数逆序与3个自然bucket完成窗口 |
| [experiments/262-data-parallel-gradient-ready-audit.md](experiments/262-data-parallel-gradient-ready-audit.md) | 两rank/多step一致性与Event overlap准入 |
| [assets/data-parallel-gradient-overlap.svg](assets/data-parallel-gradient-overlap.svg) | transient/sync/overlap total、wait与peak对比 |
| [experiments/263-data-parallel-gradient-overlap.md](experiments/263-data-parallel-gradient-overlap.md) | 1.0159×显式保留与one-process-per-GPU交接 |
| [assets/one-process-per-gpu-bootstrap.svg](assets/one-process-per-gpu-bootstrap.svg) | 独立rank/CPU reference/RCCL ID与故障传播 |
| [experiments/264-one-process-per-gpu-bootstrap.md](experiments/264-one-process-per-gpu-bootstrap.md) | 728值等价、peer终止与rank bucket准入 |
| [assets/ranked-gradient-buckets.svg](assets/ranked-gradient-buckets.svg) | collective 12×缩减与tiny启动反例 |
| [experiments/265-ranked-gradient-buckets.md](experiments/265-ranked-gradient-buckets.md) | 同步bucket baseline与Model-S交接 |
| [assets/ranked-model-s-buckets.svg](assets/ranked-model-s-buckets.svg) | Model-S collective、Reducer范围与端到端差异 |
| [experiments/266-ranked-model-s-buckets.md](experiments/266-ranked-model-s-buckets.md) | 19× collective缩减、89.3%方差与cold/steady交接 |
| [assets/ranked-steady-reducer-discard.svg](assets/ranked-steady-reducer-discard.svg) | cold/steady反转、分配/copy归因与transient拒绝 |
| [experiments/267-ranked-steady-reducer-discard.md](experiments/267-ranked-steady-reducer-discard.md) | bucket steady 0.6747×与persistent Storage准入 |
| [assets/ranked-persistent-buckets.svg](assets/ranked-persistent-buckets.svg) | 三策略steady速度、分配归零与current/peak代价 |
| [experiments/268-ranked-persistent-buckets.md](experiments/268-ranked-persistent-buckets.md) | persistent显式保留与rank gradient views准入 |
| [assets/ranked-gradient-bucket-views.svg](assets/ranked-gradient-bucket-views.svg) | 四策略速度、unpack删除与显存恢复 |
| [experiments/269-ranked-gradient-bucket-views.md](experiments/269-ranked-gradient-bucket-views.md) | views显式保留与rank ready overlap准入 |
| [assets/ranked-gradient-overlap-discard.svg](assets/ranked-gradient-overlap-discard.svg) | finish wait收益与backward enqueue抵消关系 |
| [experiments/270-ranked-gradient-overlap-discard.md](experiments/270-ranked-gradient-overlap-discard.md) | T32 total 1.0052×拒绝与尺度track交接 |
| [assets/ranked-overlap-context-scale.svg](assets/ranked-overlap-context-scale.svg) | T32/T128 finish、total、CV与显存边界 |
| [experiments/271-ranked-overlap-context-scale.md](experiments/271-ranked-overlap-context-scale.md) | T128 context-selective keep与一般默认边界 |
| [assets/ranked-checkpoint-resume.svg](assets/ranked-checkpoint-resume.svg) | rank0写、2+3恢复/5步控制与失败传播时间线 |
| [experiments/272-ranked-checkpoint-resume.md](experiments/272-ranked-checkpoint-resume.md) | 完整checkpoint字节等价与Model-S准入 |
| [assets/ranked-model-s-checkpoint.svg](assets/ranked-model-s-checkpoint.svg) | 187MB状态构成、写/等/读/恢复资源时间 |
| [experiments/273-ranked-model-s-checkpoint.md](experiments/273-ranked-model-s-checkpoint.md) | Model-S恢复等价与world-size边界交接 |
| [assets/ranked-world-size-boundary.svg](assets/ranked-world-size-boundary.svg) | world1/2成功与world4共享内存失败边界 |
| [experiments/274-ranked-world-size-boundary.md](experiments/274-ranked-world-size-boundary.md) | 一般接口保留、当前四卡能力不声明 |
| [assets/ranked-rccl-preflight.svg](assets/ranked-rccl-preflight.svg) | visible GPU、shm total/free与debug根因 |
| [experiments/275-ranked-rccl-preflight.md](experiments/275-ranked-rccl-preflight.md) | 4/4 No-space证据、unknown总需求与world2非回归 |
| [assets/ranked-input-weighting.svg](assets/ranked-input-weighting.svg) | `[B1,B2]` token交换、scale与CPU global-batch等价 |
| [experiments/276-ranked-input-weighting.md](experiments/276-ranked-input-weighting.md) | 默认拒绝、显式weighted保留与Model-S交接 |
| [assets/ranked-model-s-input-weighting.svg](assets/ranked-model-s-input-weighting.svg) | Model-S `[B1,B2]` scale、精度与显存smoke |
| [experiments/277-ranked-model-s-input-weighting.md](experiments/277-ranked-model-s-input-weighting.md) | 同步weighted保留与scale-before-ready交接 |
| [assets/ranked-weighted-overlap-discard.svg](assets/ranked-weighted-overlap-discard.svg) | finish收益、leaf scale代价与0.9594x整步反例 |
| [experiments/278-ranked-weighted-overlap-discard.md](experiments/278-ranked-weighted-overlap-discard.md) | 完整参数等价、性能拒绝与bucket weighting反驳实验 |
| [assets/ranked-bucket-weighting.svg](assets/ranked-bucket-weighting.svg) | scale 57→3、finish/整步收益与敏感性边界 |
| [experiments/279-ranked-bucket-weighting.md](experiments/279-ranked-bucket-weighting.md) | 显式T128保留、完整参数门与gather-scale交接 |
| [assets/ranked-gather-scale-discard.svg](assets/ranked-gather-scale-discard.svg) | 57 pack删除、1.0140x、额外描述与双门拒绝 |
| [experiments/280-ranked-gather-scale-discard.md](experiments/280-ranked-gather-scale-discard.md) | Step 102最佳对照与ranked reducer局部线关闭 |
| [assets/current-deepseek-t2048-profile.svg](assets/current-deepseek-t2048-profile.svg) | 当前0.8158x差距、Attention/GEMM占比与资源边界 |
| [experiments/281-current-deepseek-t2048-profile.md](experiments/281-current-deepseek-t2048-profile.md) | score/context微架构准入与allocator/KV store排除 |
| [experiments/282-cached-attention-stage-matrix.md](experiments/282-cached-attention-stage-matrix.md) | 24进程三阶段/Fused矩阵与split-sequence准入 |
| [stage timing SVG](../../benchmarks/results/2026-08-25-cached-attention-stage-matrix/stage-timing.svg) | 从24条raw生成的autoresearch风格分段图 |
| [experiments/283-cached-attention-split-search.md](experiments/283-cached-attention-split-search.md) | 144进程S1–S32搜索与官方模型门准入 |
| [split search SVG](../../benchmarks/results/2026-08-25-cached-attention-split-matrix/split-search.svg) | 八个shape的split边界与1.05门 |
| [experiments/284-cached-attention-split-model-reject.md](experiments/284-cached-attention-split-model-reject.md) | 2.222x速度通过但完整logits拒绝 |
| [split model SVG](../../benchmarks/results/2026-08-25-cached-attention-split-model/comparison.svg) | 三对吞吐、精度与资源结果 |
| [experiments/285-materialized-score-attention.md](experiments/285-materialized-score-attention.md) | 位级保序并行QK的24进程算子准入 |
| [materialized score SVG](../../benchmarks/results/2026-08-25-cached-attention-materialized-matrix/comparison.svg) | 八格current/materialized Event与score bytes |
| [experiments/286-materialized-score-model.md](experiments/286-materialized-score-model.md) | DeepSeek位级相同、1.3207x模型保留 |
| [materialized model SVG](../../benchmarks/results/2026-08-25-cached-attention-materialized-model/comparison.svg) | 三对吞吐、完整精度与资源变化 |
| [experiments/287-materialized-score-model-boundary.md](experiments/287-materialized-score-model-boundary.md) | Qwen/DeepSeek默认minimum=2048的八格证据 |
| [materialized boundary SVG](../../benchmarks/results/2026-08-25-materialized-attention-model-matrix/matrix.svg) | 两模型、两context、两batch性能与精度门 |
| [experiments/288-materialized-score-auto-policy.md](experiments/288-materialized-score-auto-policy.md) | 用户默认auto与显式off的四格因果复测 |
| [auto policy SVG](../../benchmarks/results/2026-08-25-materialized-attention-auto-matrix/matrix.svg) | Qwen/DeepSeek T2048默认收益 |
| [experiments/289-post-materialized-profile.md](experiments/289-post-materialized-profile.md) | 新默认后的阶段迁移、finalize热点与allocator反例 |
| [post-materialized profile SVG](../../benchmarks/results/2026-08-25-post-materialized-deepseek-t2048-profile/profile-delta.svg) | 自动生成的Kernel阶段占比图 |
| [experiments/290-finalize-thread-mapping-discard.md](experiments/290-finalize-thread-mapping-discard.md) | 96进程保序线程映射反例与P×V交接 |
| [finalize mapping SVG](../../benchmarks/results/2026-08-25-cached-attention-finalize-mapping/mapping.svg) | 64/128相对256线程的16格速度图 |
| [experiments/291-exact-softmax-split-pv.md](experiments/291-exact-softmax-split-pv.md) | exact softmax下隔离P×V并行度的160进程算子准入 |
| [split-PV search SVG](../../benchmarks/results/2026-08-25-cached-attention-split-pv-matrix/split-pv-search.svg) | S1反例与16格S16 winner |
| [experiments/292-split-pv-model-reject.md](experiments/292-split-pv-model-reject.md) | 1.4834x但完整logits失败，定位到P×V加法树 |
| [split-PV model SVG](../../benchmarks/results/2026-08-25-cached-attention-split-pv-model/comparison.svg) | 三对速度、完整分布与资源反例 |
| [experiments/293-exact-gqa-value-reuse-reject.md](experiments/293-exact-gqa-value-reuse-reject.md) | 位级GQA复用0/16过门与exact-finalize局部线关闭 |
| [GQA value-reuse SVG](../../benchmarks/results/2026-08-25-cached-attention-gqa-value-reuse/value-reuse.svg) | 128进程tile矩阵与全局probability代价 |
| [experiments/294-serving-batch-scale.md](experiments/294-serving-batch-scale.md) | T2048 B1/2/4/8吞吐、效率、显存与token边界 |
| [serving batch SVG](../../benchmarks/results/2026-08-25-serving-batch-scale/batch-scale.svg) | micro/PyTorch batch扩展与每请求资源 |
| [experiments/295-cross-batch-logit-audit.md](experiments/295-cross-batch-logit-audit.md) | DeepSeek内部batch漂移、行/argmax排除与精度隔离交接 |
| [cross-batch SVG](../../benchmarks/results/2026-08-25-deepseek-cross-batch-logits/cross-batch.svg) | step0/1/2完整logits Max/RMS |
| [experiments/296-cross-batch-precision-isolation.md](experiments/296-cross-batch-precision-isolation.md) | 四精度岛隔离与BF16 FFN block trace准入 |
| [precision-isolation SVG](../../benchmarks/results/2026-08-25-deepseek-cross-batch-precision/precision-isolation.svg) | FP32/FFN/Attention/both完整logits漂移 |
| [experiments/297-cached-block-drift.md](experiments/297-cached-block-drift.md) | 31个边界定位到Block 0首次放大，默认策略冻结 |
| [cached block-drift SVG](../../benchmarks/results/2026-08-25-deepseek-cached-block-drift/block-drift.svg) | FP32 Linear与BF16 FFN-only逐层Max误差 |
| [experiments/298-cached-block-zero-detail.md](experiments/298-cached-block-zero-detail.md) | BF16 input/gate/down逐点放大与单层FP32反驳实验 |
| [cached block-detail SVG](../../benchmarks/results/2026-08-26-deepseek-cached-block-detail/block-detail.svg) | Block 0内部relative-L2对数图 |
| [experiments/299-block-zero-fp32-reject.md](experiments/299-block-zero-fp32-reject.md) | 单层FP32的batch反例、82.6MB代价与algorithm交接 |
| [block-0 counterfactual SVG](../../benchmarks/results/2026-08-26-deepseek-bf16-ffn-layer-counterfactual/counterfactual.svg) | FP32/全BF16/Block0-FP32完整logits |
| [experiments/300-bf16-decode-algorithm-reject.md](experiments/300-bf16-decode-algorithm-reject.md) | 64候选交集、75892反例与算子row-invariance交接 |
| [BF16 decode algorithm SVG](../../benchmarks/results/2026-08-26-deepseek-bf16-decode-algorithm/algorithm.svg) | 默认与固定solution的完整logits/吞吐边界 |
| [experiments/301-bf16-row-invariance.md](experiments/301-bf16-row-invariance.md) | 64/64相同BF16输入跨M exact与gate/up track关闭 |
| [BF16 row-invariance SVG](../../benchmarks/results/2026-08-26-bf16-decode-row-invariance/row-invariance.svg) | 64个solution的CPU/row双门 |
| [experiments/302-prefill-cache-prefix.md](experiments/302-prefill-cache-prefix.md) | decode前K/V已漂移与prefill QKV trace交接 |
| [prefill cache-prefix SVG](../../benchmarks/results/2026-08-26-deepseek-prefill-cache-prefix/cache-prefix.svg) | Block0完整BF16 K/V跨batch误差 |
| [experiments/303-prefill-block0-trace.md](experiments/303-prefill-block0-trace.md) | FP32 Q projection首次非零与大M solution交接 |
| [prefill block-0 trace SVG](../../benchmarks/results/2026-08-26-deepseek-prefill-block0-trace/prefill-trace.svg) | 10个边界的B2/B4/B8完整row误差 |
| [experiments/304-fp32-qkv-row-invariance.md](experiments/304-fp32-qkv-row-invariance.md) | Q/KV跨M位级solution与完整模型准入 |
| [FP32 QKV row-invariance SVG](../../benchmarks/results/2026-08-26-fp32-qkv-row-invariance/qkv-row-invariance.svg) | Q 12选1、KV 22选5的双门图 |
| [experiments/305-fp32-qkv-model-reject.md](experiments/305-fp32-qkv-model-reject.md) | Cache exact但logits/性能拒绝与post-cache trace交接 |
| [FP32 QKV model-gate SVG](../../benchmarks/results/2026-08-26-fp32-qkv-model-gate/model-gate.svg) | raw cache、完整logits和无导出性能 |
| [experiments/306-post-cache-block0-trace.md](experiments/306-post-cache-block0-trace.md) | exact cache后的Attention context首差与core拆解 |
| [post-cache trace SVG](../../benchmarks/results/2026-08-26-post-cache-block0-trace/post-cache-trace.svg) | 17个Block0边界的跨/内batch误差 |
| [experiments/307-prefill-attention-core-diagnostics.md](experiments/307-prefill-attention-core-diagnostics.md) | opt-in QK/softmax/P×V诊断边界与默认路径非侵入门 |
| [Attention-core diagnostics SVG](../../benchmarks/results/2026-08-26-prefill-attention-core-diagnostics/diagnostics.svg) | 生产/诊断双路径与五组回归证据 |
| [experiments/308-prefill-attention-core-matrix.md](experiments/308-prefill-attention-core-matrix.md) | B2 P×V与B4/B8 QK双首差、causal-visible完整值门 |
| [Attention-core matrix SVG](../../benchmarks/results/2026-08-26-prefill-attention-core-matrix/attention-core.svg) | 8进程三阶段完整二进制比较 |
| [experiments/309-fp32-attention-batch-solutions.md](experiments/309-fp32-attention-batch-solutions.md) | QK/P×V exact能力与四batch性能拒绝 |
| [Attention solution SVG](../../benchmarks/results/2026-08-26-fp32-attention-batch-invariance/attention-solutions.svg) | 34+2候选的exact/最差speedup矩阵 |
| [experiments/310-prefill-attention-model-reject.md](experiments/310-prefill-attention-model-reject.md) | core exact但完整logits/性能拒绝 |
| [Attention model-gate SVG](../../benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/model-gate.svg) | 32进程完整core/logits/prefill门 |
| [experiments/311-batch-selective-attention-reject.md](experiments/311-batch-selective-attention-reject.md) | 性能/RMS通过但Max/B2拒绝与路线关闭 |
| [Selective gate SVG](../../benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate/selective-gate.svg) | batch-local index的完整logits/性能矩阵 |
| [experiments/312-post-exact-core-o-projection.md](experiments/312-post-exact-core-o-projection.md) | exact context后的O projection首差 |
| [Post-exact-core SVG](../../benchmarks/results/2026-08-26-post-exact-core-block0-trace/post-exact-core-trace.svg) | 8进程17边界完整trace |
| [experiments/313-post-exact-o-ffn-output.md](experiments/313-post-exact-o-ffn-output.md) | O exact后的FFN output首差 |
| [Post-exact-O SVG](../../benchmarks/results/2026-08-26-post-exact-o-block0-trace/post-exact-o-trace.svg) | 8进程O/FFN边界trace |
| [experiments/314-prefill-o-model-reject.md](experiments/314-prefill-o-model-reject.md) | O完整logits改善但B1性能失败的拒绝证据 |
| [O model-gate SVG](../../benchmarks/results/2026-08-26-fp32-prefill-o-model-gate/o-model-gate.svg) | 16+16进程数值、prefill和资源门 |
| [experiments/315-prefill-exact-stack-reject.md](experiments/315-prefill-exact-stack-reject.md) | 真实upstream下exact组合数值拒绝与路线关闭 |
| [Exact-stack SVG](../../benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate/exact-stack-gate.svg) | Release 16+16进程完整模型门 |
| [FFN detail-trace SVG](assets/prefill-ffn-detail-trace.svg) | gate/up/SwiGLU/down显式两行数值显微镜 |
| [experiments/316-prefill-ffn-gate-up.md](experiments/316-prefill-ffn-gate-up.md) | norm exact、gate/up双投影首差与下一descriptor |
| [FFN stage-trace SVG](../../benchmarks/results/2026-08-26-prefill-ffn-stage-trace/ffn-stage-trace.svg) | 8进程7阶段完整值误差图 |
| [experiments/317-fp32-ffn-solution-reject.md](experiments/317-fp32-ffn-solution-reject.md) | 唯一exact index的M8192性能反例 |
| [FFN row-invariance SVG](../../benchmarks/results/2026-08-26-fp32-ffn-row-invariance/ffn-row-invariance.svg) | 33候选correctness-before-timing矩阵 |
| [experiments/318-prefill-ffn-selective-reject.md](experiments/318-prefill-ffn-selective-reject.md) | 性能/Max过门但RMS失败的整模反例 |
| [FFN model-gate SVG](../../benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate/ffn-model-gate.svg) | 32进程真实upstream整模门 |
| [experiments/319-prefill-ffn-all-exact-reject.md](experiments/319-prefill-ffn-all-exact-reject.md) | Max改善但RMS恶化与vendor模型线关闭 |
| [All-exact FFN SVG](../../benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate/ffn-all-exact-model-gate.svg) | all-batch 32进程最终反驳 |
| [experiments/320-post-exact-gate-up-down.md](experiments/320-post-exact-gate-up-down.md) | gate/up/SwiGLU exact与down首差 |
| [Post-exact gate/up SVG](../../benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace/post-exact-gate-up-trace.svg) | 8进程7阶段完整值因果图 |
| [experiments/321-fp32-ffn-down-reject.md](experiments/321-fp32-ffn-down-reject.md) | down唯一exact候选的0.506×性能反例 |
| [FFN down SVG](../../benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance/ffn-down-row-invariance.svg) | K8960/N1536的15候选矩阵 |
| [experiments/322-clean-long-context-profile.md](experiments/322-clean-long-context-profile.md) | 当前1.1393×PyTorch与42.27% finalize热点 |
| [Current clean profile SVG](../../benchmarks/results/2026-08-26-clean-deepseek-t2048-profile/profile-delta.svg) | 1/3-generation当前Kernel分类 |
| [experiments/323-finalize-architecture-gap.md](experiments/323-finalize-architecture-gap.md) | 原生128与旧logical-256 mapping的结构差异 |
| [Finalize gap SVG](../../benchmarks/results/2026-08-26-finalize-architecture-gap-audit/finalize-gap.svg) | 当前/旧mapping/native128架构对照 |
| [experiments/324-native128-finalize-reject.md](experiments/324-native128-finalize-reject.md) | accuracy通过但T2048约1.003×的反例 |
| [Native128 SVG](../../benchmarks/results/2026-08-26-native128-finalize/native128.svg) | 16进程Event/wall/精度矩阵 |
| [experiments/325-grouped-gate-up-row2.md](experiments/325-grouped-gate-up-row2.md) | 当前decode rows2 grouped GEMM空白 |
| [Grouped rows2 SVG](../../benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2/grouped-row2.svg) | 三轮Event/wall与稳定性 |
| [experiments/326-grouped-decode-model-reject.md](experiments/326-grouped-decode-model-reject.md) | 1.00968×未过1.01模型门 |
| [experiments/327-clean-deepseek-local-saturation.md](experiments/327-clean-deepseek-local-saturation.md) | 当前固定workload局部停止门 |
| [experiments/328-external-gradient-pool-discard.md](experiments/328-external-gradient-pool-discard.md) | 全梯度exact、地址稳定但默认性能拒绝 |
| [External gradient pool SVG](assets/external-gradient-pool-discard.svg) | 18进程Event/wall/峰值决策图 |
| [experiments/329-pytorch-rocm-custom-ops.md](experiments/329-pytorch-rocm-custom-ops.md) | ROCm dispatcher正确、elementwise性能拒绝 |
| [PyTorch ROCm Custom Ops SVG](assets/pytorch-rocm-custom-ops.svg) | 三dtype与前反向代表格 |
| [experiments/330-pytorch-custom-op-vector16.md](experiments/330-pytorch-custom-op-vector16.md) | broad反例与selective keep |
| [Custom Op vector16 SVG](assets/pytorch-rocm-custom-op-vector16.svg) | scalar/broad/selective 16M对比 |
| [experiments/331-pytorch-custom-op-swiglu.md](experiments/331-pytorch-custom-op-swiglu.md) | 融合forward成功、training反例 |
| [Fused SwiGLU SVG](assets/pytorch-rocm-custom-op-swiglu.svg) | 三dtype forward/backward边界 |
| [experiments/332-pytorch-swiglu-backward-vector-reject.md](experiments/332-pytorch-swiglu-backward-vector-reject.md) | float4解释被推翻并清理 |
| [SwiGLU backward SVG](assets/pytorch-rocm-swiglu-backward.svg) | scalar/vector/native producer对比 |
| [experiments/333-pytorch-swiglu-scalar-seed.md](experiments/333-pytorch-swiglu-scalar-seed.md) | 4-byte seed不再物化完整Tensor |
| [Scalar-seed SVG](assets/pytorch-rocm-swiglu-scalar-seed.svg) | F+B时间与peak前后对比 |
| [experiments/334-pytorch-swiglu-autograd-attribution.md](experiments/334-pytorch-swiglu-autograd-attribution.md) | 剩余差距定位到Autograd提交 |
| [Autograd attribution SVG](assets/pytorch-rocm-swiglu-autograd-attribution.svg) | native/custom/manual三路Event |
| [experiments/335-pytorch-swiglu-compile-reject.md](experiments/335-pytorch-swiglu-compile-reject.md) | compiled比eager更慢且有冷启动 |
| [SwiGLU compile SVG](assets/pytorch-rocm-swiglu-compile.svg) | 四路steady/cold结果 |
| [experiments/336-pytorch-swiglu-cpp-autograd.md](experiments/336-pytorch-swiglu-cpp-autograd.md) | C++ callback关闭主要gap并推荐 |
| [C++ Autograd SVG](assets/pytorch-rocm-swiglu-cpp-autograd.svg) | 三dtype速度/peak准入 |
| [experiments/337-pytorch-swiglu-typed-backward.md](experiments/337-pytorch-swiglu-typed-backward.md) | 低精度F+B达到scoped parity |
| [Typed backward SVG](assets/pytorch-rocm-swiglu-typed-backward.svg) | FP16/BF16 ATen/native准入 |
| [experiments/338-pytorch-typed-softmax-baseline.md](experiments/338-pytorch-typed-softmax-baseline.md) | 无FP32临时量但serial性能失败 |
| [Typed Softmax SVG](assets/pytorch-rocm-typed-softmax.svg) | 10格log-scale性能图 |
| [experiments/339-pytorch-block-softmax.md](experiments/339-pytorch-block-softmax.md) | block归约消除模型宽度串行悬崖 |
| [Block Softmax SVG](assets/pytorch-rocm-block-softmax.svg) | serial/block/PyTorch十格对比 |
| [experiments/340-pytorch-cached-softmax.md](experiments/340-pytorch-cached-softmax.md) | bounded LDS cache删除第二次exp |
| [Cached Softmax SVG](assets/pytorch-rocm-cached-softmax.svg) | width4096 Torch/block/cache Event图 |
| [experiments/341-pytorch-wave-softmax-reject.md](experiments/341-pytorch-wave-softmax-reject.md) | broad wave归约因BF16 wall失败而删除 |
| [Wave Softmax reject SVG](assets/pytorch-rocm-wave-softmax-reject.svg) | 两dtype Event/wall与1.05门 |
| [experiments/342-pytorch-fp16-wave-softmax.md](experiments/342-pytorch-fp16-wave-softmax.md) | FP16-only wave谓词通过 |
| [FP16 wave Softmax SVG](assets/pytorch-rocm-fp16-wave-softmax.svg) | selective/fallback Event与wall |
| [experiments/343-pytorch-fast-exp-softmax-reject.md](experiments/343-pytorch-fast-exp-softmax-reject.md) | fast-exp正确但性能不足并删除 |
| [Fast-exp Softmax reject SVG](assets/pytorch-rocm-fast-exp-softmax-reject.svg) | Event/wall与1.05门 |
| [experiments/344-pytorch-softmax-thread-matrix.md](experiments/344-pytorch-softmax-thread-matrix.md) | 128/256/512/1024线程矩阵选择1024 |
| [Softmax thread matrix SVG](assets/pytorch-rocm-softmax-thread-matrix.svg) | 四档Event/wall与PyTorch线 |
| [experiments/345-pytorch-softmax-attribution.md](experiments/345-pytorch-softmax-attribution.md) | raw/C++/Python/PyTorch分层归因 |
| [Softmax attribution SVG](assets/pytorch-rocm-softmax-attribution.svg) | 四层Event中位数与边界比值 |
| [experiments/346-pytorch-custom-op-softmax.md](experiments/346-pytorch-custom-op-softmax.md) | C++ Custom Op与no-grad门 |
| [Custom Op Softmax SVG](assets/pytorch-rocm-custom-op-softmax.svg) | 十格native比与wide反例 |
| [experiments/347-pytorch-custom-op-softmax-out.md](experiments/347-pytorch-custom-op-softmax-out.md) | caller-owned mutation/alias合同 |
| [Custom Op Softmax out SVG](assets/pytorch-rocm-custom-op-softmax-out.svg) | native out十格与零peak |
| [experiments/348-pytorch-bf16-wave1024-softmax.md](experiments/348-pytorch-bf16-wave1024-softmax.md) | BF16 1024-thread wave反驳旧256结论 |
| [BF16 wave1024 SVG](assets/pytorch-rocm-bf16-wave1024-softmax.svg) | core/Custom out前后Event |
| [experiments/349-pytorch-softmax-out-fallthrough-reject.md](experiments/349-pytorch-softmax-out-fallthrough-reject.md) | Autograd fallthrough无收益并删除 |
| [Softmax out fallthrough SVG](assets/pytorch-rocm-softmax-out-fallthrough-reject.svg) | 两dtype相对1.05门 |
| [experiments/350-official-pytorch-hidden-alignment.md](experiments/350-official-pytorch-hidden-alignment.md) | 真实Qwen/DeepSeek逐层PyTorch对齐 |
| [Official hidden alignment SVG](assets/official-pytorch-hidden-alignment.svg) | 27/31阶段relative-L2曲线 |
| [experiments/351-official-hf-fixtures.md](experiments/351-official-hf-fixtures.md) | pinned官方权重/tokenizer fixture |
| [Official HF fixtures SVG](assets/official-hf-fixtures.svg) | 权重规模与结构/许可门 |
| [experiments/352-int8-weight-contract.md](experiments/352-int8-weight-contract.md) | 一字节Tensor、显式scale与跨框架weight-only合同 |
| [INT8 weight contract SVG](assets/int8-weight-contract.svg) | 存储缩减、四类正确性门与未接模型边界 |
| [experiments/353-int8-weight-matmul-baseline.md](experiments/353-int8-weight-matmul-baseline.md) | 完整反量化+matmul正确性和allocation基线 |
| [INT8 matmul baseline SVG](assets/int8-weight-matmul-baseline.svg) | 从I8权重到完整浮点临时量的公开代价 |
| [experiments/354-int8-fused-decode.md](experiments/354-int8-fused-decode.md) | M=1融合INT8读取、跨框架速度与resident-GEMM反例 |
| [INT8 fused decode SVG](assets/int8-fused-decode.svg) | dequant收益、临时内存与DeepSeek反例 |
| [experiments/355-model-s-int8-inference.md](experiments/355-model-s-int8-inference.md) | Model-S整模M=1收益与M>1反例 |
| [Model-S INT8 SVG](assets/model-s-int8-boundary.svg) | throughput、常驻与事务准备峰值 |
| [experiments/356-official-int8-device-amax-reject.md](experiments/356-official-int8-device-amax-reject.md) | device准备成功与官方Qwen精度拒绝 |
| [Official INT8 reject SVG](assets/official-int8-device-amax-reject.svg) | 准备/显存收益和完整logits反例 |
| [experiments/357-official-int8-column-reject.md](experiments/357-official-int8-column-reject.md) | 逐输出通道改善与最终官方精度拒绝 |
| [INT8 column reject SVG](assets/official-int8-column-reject.svg) | 完整logits和token停止门 |
| [experiments/358-official-int8-scope-matrix.md](experiments/358-official-int8-scope-matrix.md) | FFN/Attention混合精度隔离 |
| [INT8 scope SVG](assets/official-int8-scope-matrix.svg) | 固定Max/RMS/token门 |
| [experiments/359-official-int8-attention-split-saturation.md](experiments/359-official-int8-attention-split-saturation.md) | QKV/O最终拆分与饱和停止 |
| [Final INT8 split SVG](assets/official-int8-attention-split-saturation.svg) | 不事后放宽门槛的拒绝证据 |
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
