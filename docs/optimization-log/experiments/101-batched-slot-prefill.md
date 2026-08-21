# Experiment 101 — decode已经并行，prefill也不能逐个排队

uniform continuous在Experiment 098只有static batch约31%，clean trace显示每次仍有8个row-prefill。
本节点把同长度pending prompt合成一次`[A,T]`模型prefill，再映射到共享Cache目标rows。

## 合同

- active rows严格递增、目标必须为空；
- 相同prompt长度才允许成组；
- 最早pending请求决定下一组长度，同长度请求保持提交顺序；
- 不同长度在同step进入独立组，不用padding伪装兼容；
- logical prefill rows、请求数和生成结果不能减少；
- existing Cache row完整内容保持；
- FP32/BF16、CPU/HIP通过。

![Batched slot prefill](../assets/batched-slot-prefill.svg)

## 物理调用证据

| shape | logical rows | prefill batches | batched calls | rows in batches |
|---|---:|---:|---:|---:|
| uniform R8/S8 | 8 | 1 | 1 | 8 |
| R8/S4 | 8 | 5 | 3 | 6 |
| R8/S2 | 8 | 7 | 1 | 2 |

这些数解释了性能梯度，而不是只给出一条漂亮结果。

## 严格交替Release A/B

| shape | baseline median | candidate median | speedup | 额外结论 |
|---|---:|---:|---:|---|
| uniform R8/S8 | 5418.05 | 15880.60 | 2.931× | static的87.4% |
| R8/S4 | 3829.64 | 5029.89 | 1.313× | 6 rows真正合批 |
| R8/S2 | 2733.47 | 2887.69 | 1.056× | 仅2 rows合批 |

9/9逐对candidate更快，18个进程输出和checksum一致，reference漂移在±1%附近。收益随真正合批行数
8→6→2而减弱，支持“逐row prefill是uniform主要缺口”的解释。

R8/S4 active Cache峰值可能增加，因为更快的分组让更多真实请求同时驻留；allocated Cache固定不变。
报告把它记录为并发驻留代价，不当成内存泄漏。

下一步可研究不同长度prompt的padding/packed prefill，但必须同时计算无效token成本；不能为了减少
call数就把大量padding藏起来。

数据见 [`101-data`](101-data/)。
