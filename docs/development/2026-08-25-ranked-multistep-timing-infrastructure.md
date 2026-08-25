# 2026-08-25 — ranked multi-step cold/steady timing infrastructure

Experiment 266的一步Reducer方差达到89.3%，说明一次进程总计时仍把首次HIP/hipBLASLt/RCCL建立
成本与steady训练混在一起。本节点不删除cold数据，而是为每个rank逐step记录：

```text
forward + backward
reducer
optimizer
complete step
```

Reducer区间还记录每步collective、bucket、pack/unpack、logical allocation、backend allocation、
deallocation和total allocated bytes。launcher先校验两rank数组长度、有限性、阶段求和与总计数，
再用较慢rank形成每步组指标。matrix用显式`--steady-skip-steps`把第1步标记为cold，保留其原值，
后续样本单独计算中位数、范围和CV。

## Pilot

Model-S、两rank、`B1×T32`、3步、25 MiB的单次pilot显示：

- bucket forward/backward：cold约5.73s，steady约5.97/6.37ms；
- bucket Reducer：cold约112.96ms，steady约4.56/4.58ms；
- 每个bucket steady step仍有60次backend allocation和124,689,408 bytes临时分配；
- 每步仍有57次pack与57次unpack；
- per-parameter steady Reducer约2.66/2.79ms，且Reducer分配为0；
- 三步后rank仍exact，CPU Max/RMS为`0.0062715/3.701e-6`，loss差`1.967e-5`。

单次pilot提示当前transient bucket可能在steady状态更慢，但不能替代三次交错正式矩阵。下一提交
从已push的干净revision运行正式Experiment 267；如果反例稳定，就先拒绝transient bucket作为
性能路线，再用persistent rank plan单独消除60次分配和114次copy。

基础设施回归：完整RCCL标签43/43，`DistributedRank.*` 5/5，optimization log验证通过，
coverage audit仍为175个算子、42个graph API和123个测试文件。
