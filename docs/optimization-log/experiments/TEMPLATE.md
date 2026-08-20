# Experiment NNN — short name

Status: `planned | running | keep | discard | crash | invalid`

## Observed bottleneck

写 profiler 数字和源码位置，不写“感觉这里慢”。

## Hypothesis

只写一个主要因果假设，以及哪个结果会推翻它。

## Scope

- allowed files:
- public API changes:
- forbidden changes:
- ownership/stream changes:

## Fixed comparison

```text
GPU:
ROCm:
microLLM commit:
PyTorch/Transformers:
dtype:
models/tokens:
warm-up:
measured:
```

## Correctness gates

- [ ] focused CPU test
- [ ] focused HIP test
- [ ] PyTorch forward/backward comparison
- [ ] Qwen exact tokens/loss trajectory
- [ ] DeepSeek exact tokens/loss trajectory
- [ ] full CPU/sanitizer/HIP regression

## Commands

```bash
# exact commands go here
```

## Results

| Workload | Before | After | Ratio delta | Peak memory delta |
|---|---:|---:|---:|---:|
| Qwen train | | | | |
| Qwen generate | | | | |
| DeepSeek train | | | | |
| DeepSeek generate | | | | |

## Profiler evidence

保留 before/after 的热点、API calls、copy、allocation 和 launch 数量。

## Decision

`keep | discard | crash | invalid`

说明证据支持了什么、没有支持什么，以及下一次实验不能同时改变什么。
