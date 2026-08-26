# 2026-08-26 — native128 finalize operator与矩阵

新增default-off research operator。它使用128-entry reduction、stride128和128-column P×V，不修改当前
256路径或Auto策略。CPU回退、MI300X完整输出、FP32/BF16 cache和非法layout都有测试。

矩阵固定DeepSeek T512/T2048、B1/B2、FP32/BF16 cache、两个fresh process，共16行。每行先检查完整
输出Max/RMS/finite，再对同进程materialized256测Event/wall；T2048四个case都必须Event≥1.05×且
wall≥1.02×。失败后candidate源码和runner删除，证据保留。
