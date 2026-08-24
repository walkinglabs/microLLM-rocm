# Experiment 208 — BF16 V cast+repeat 融合被 batch 反例拒绝

Status: explicit primitive retained; model integration cancelled

## 假设与实现

Grouped V先输出BF16，旧模型边界执行一次BF16→FP32 cast，再按GQA head重复。候选Kernel直接
读取BF16并写出重复后的FP32 Tensor，数学上等价于：

```text
repeat_interleave(ops::cast(value_bf16, FP32), head_dim, repeats)
```

新增`repeat_interleave_bf16_to_float`同时有CPU reference、HIP typed Kernel、零payload-transfer
测试和外部CMake consumer链接门。它不改变Attention dtype，也没有接进模型。

## 三进程算子矩阵

![BF16 repeat fusion discard](../assets/bf16-repeat-fusion-discard.svg)

| Family | B1/T256 | B1/T512 | B1/T1024 | B2/T512 |
|---|---:|---:|---:|---:|
| Qwen | 1.253× | 1.291× | 0.996× | 1.004× |
| DeepSeek | 1.345× | 1.027× | 1.011× | 0.995× |

48个进程全部完整输出逐项相等且计时无H2D/D2H。只有3/8通过1.05算子门，两条B2都失败。
这说明少一次launch对小B1有价值，但输出规模增大后主要成本是写完整expanded V，省掉较小的
cast不足以改变总时间。

## 过程纠错

第一次benchmark错误使用`Tensor::cast`，每轮产生20次H2D/D2H并被transfer门立即拒绝。改成
设备原生`ops::cast`后重新从头运行，正式数据不包含该无效pilot。这个错误本身保留在开发记录，
避免把host fallback当作旧模型基线。

## 决策

不接模型，不添加CLI策略。显式primitive和benchmark保留，供小B1专用调度或其他GPU研究；
Auto路径不变。下一候选不能只删除一个小中间Tensor，必须处理expanded V本身或概率物化。

原始证据：[operator matrix](../../../benchmarks/results/2026-08-24-bf16-repeat-operator/)。
