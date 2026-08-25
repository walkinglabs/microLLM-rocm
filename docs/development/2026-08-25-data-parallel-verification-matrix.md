# 2026-08-25 — 参数审计频率矩阵

![Data parallel verification interval](../optimization-log/assets/data-parallel-verification-interval.svg)

9个新进程的180个loss完全相同。每步审计steady 2.96ms；只在最终step审计为2.38ms；关闭为
2.52ms。默认仍是每步检查，性能运行必须显式选择interval。

首版skip暴露了optimizer隐式依赖host copy等待的问题。现在optimizer阶段显式同步，所以
interval只控制审计。下一步用final-step检查扫描真实bucket count。

