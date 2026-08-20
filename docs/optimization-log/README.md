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
