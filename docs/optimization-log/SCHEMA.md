# Optimization experiment contract

## 固定环境

第一阶段固定：

```text
GPU             AMD Instinct MI300X VF
architecture    gfx942:sramecc+:xnack-
microLLM dtype  FP32
PyTorch dtype   FP32
warm-up         2 complete iterations
measured        5 complete iterations
```

如果 GPU、ROCm、dtype、模型 revision、token、batch、context、warm-up 或 measured
次数变化，必须建立新的 track，不能与当前 running best 连线。

## 主分数

四项 throughput ratio 都是：

```text
microLLM tokens/s ÷ PyTorch tokens/s
```

主分数：

```text
score = (qwen_train × qwen_generate × deepseek_train × deepseek_generate)^(1/4)
```

使用几何平均是因为四项都是比值；任意一项接近 0 都会显著拉低总分。

对于预期收益小于 10% 的候选，四项 ratio 使用至少三次独立进程的逐 workload
中位数；每个进程内部仍是固定 warm-up/measured。一次进程的高点或低点不能独自决定
keep/discard。

改变 dtype 的实验不能追加到 FP32 `results.tsv` running-best 曲线。它们使用独立结果表
和图，例如 `bf16-results.tsv`/`bf16-gemm.svg`，直到同 dtype 的完整 PyTorch/model
矩阵建立后再定义该 track 的端到端 score。

## 保留门

一次实验只有同时满足以下条件才可标为 `keep`：

1. CPU、sanitizer、HIP 和相关 PyTorch oracle 通过；
2. Qwen/DeepSeek exact greedy token 不变；
3. loss/gradient/parameter update 在既定容差内；
4. 主分数高于当前 running best；
5. 任一 workload 不得无解释地退化超过 5%；
6. 显存上升必须记录绝对值、比例和原因；
7. 未删除 reference、未加入全局同步、未改变测量协议；
8. diff 仍能由贡献者解释和维护。

## results.tsv

列定义：

```text
experiment       从 000 开始的三位编号
commit           实验 commit 的短标识
date             UTC 日期
status           baseline/keep/discard/crash/invalid
score            四项几何平均；失败使用 0
qwen_train       microLLM/PyTorch throughput ratio
qwen_generate    microLLM/PyTorch throughput ratio
deepseek_train   microLLM/PyTorch throughput ratio
deepseek_generate microLLM/PyTorch throughput ratio
qwen_train_mem   microLLM engine peak / PyTorch allocated peak
qwen_generate_mem 同上
deepseek_train_mem 同上
deepseek_generate_mem 同上
description      不含 tab 的单行说明
```

实验编号在所有 dtype/track 间全局递增，因此 FP32 `results.tsv` 可以出现空号；例如
014–015 属于独立 BF16 track。表内编号必须严格递增且不得重复，但不要求连续。

`*_mem` 保持历史可比性，使用同一时刻的逻辑活跃 Tensor peak。启用 caching allocator
后，实验报告和 raw JSONL 还必须单独给出 `cached_bytes` 与 `reserved_bytes`；pool
保留的物理内存不能藏在逻辑 peak 后面，也不能混入旧行后重写历史分数。

`commit` 是 Git commit 标识，不是文件内容校验值。

## 每个实验必须保存

- 任务合同和被改变的唯一主要变量；
- 完整命令；
- 系统、编译器、ROCm 和 GPU 元数据；
- correctness 输出；
- microLLM raw JSONL；
- PyTorch raw JSONL 或固定参考文件；
- 自动 comparison JSONL；
- 必要时 rocprof CSV/PFTrace 的保留位置；
- 结果解释、未支持结论和 keep/discard 决策。

大体积 PFTrace 可以作为 release artifact，不强制提交 Git；报告必须记录可获取位置。
