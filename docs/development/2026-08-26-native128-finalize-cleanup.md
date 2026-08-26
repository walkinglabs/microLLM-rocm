# 2026-08-26 — 删除失败的native128 finalizer

Experiment 324性能失败后，本提交删除native128 public research API、HIP Kernel、benchmark flag、formal
runner和candidate-specific测试。当前materialized256、Auto策略、raw矩阵、SVG与实验文档全部保留。

状态合同检查runner文件不存在，且当前include/src/benchmark/tests不含native128符号，防止约1.003×的
失败路径重新进入代码。
