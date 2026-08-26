# Step 138 — Refresh clean-upstream long-context baseline and profile

Status: completed by Experiment 322

数值row-order线已关闭，且所有失败模型route已删除。下一节点从clean Release revision重跑：

- DeepSeek T2048、B2、N64 cached decode；
- microLLM与同环境PyTorch ROCm，64 token精确、resident/peak/KV对齐；
- 至少两个fresh process或原合同的成对测量；
- 当前microLLM rocprof phase-delta，重新统计Attention、GEMM、cast、allocator和其他Kernel；
- 不沿用旧61.57% Attention占比作为当前结论。

只有新profile的最大热点可以进入下一优化节点。

结果：三轮micro/PyTorch中位数177.77/156.04 tok/s，即1.1393×，64 token exact，峰值5.23/6.38GB。
当前Kernel 820.74ms，finalize 42.27%、GEMM 33.25%、scores 7.88%。下一步只审计新的finalize架构。
