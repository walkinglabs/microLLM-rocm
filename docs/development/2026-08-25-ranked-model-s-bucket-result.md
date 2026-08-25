# 2026-08-25 — ranked Model-S natural-bucket result

![Ranked Model-S buckets](../optimization-log/assets/ranked-model-s-buckets.svg)

Model-S `B1×T32/rank`的一步正式矩阵把collective/rank从57降到3。三次fresh进程中位数显示
Reducer `54.51→32.48ms`，但bucket范围是`19.55–158.52ms`、CV 89.3%；完整训练与组wall
仅改善1.0016×/1.0023×。因此保留为正确性与测量baseline，不作稳定性能结论。

57个Tensor、15,586,176个值跨rank exact；CPU Max/RMS为`0.0062738/3.483e-6`；loss差、
peer failure和临时文件清理全部通过。下一节点做多步逐step cold/steady分离，再决定persistent
rank bucket。
