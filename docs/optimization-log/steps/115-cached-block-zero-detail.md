# Step 115 — Cached block-0 internal drift

Status: planned

Experiment 297已经证明Embedding位级相同，而Batch漂移第一次出现在Block 0完整输出。这个节点只
打开Block 0细节，不测性能，也不改变默认策略。

固定DeepSeek、T2048、step0、B1/B2、FP32 Linear与BF16 FFN-only、两个fresh process。依次比较：

- attention norm、Q/K/V与attention output；
- attention residual和FFN norm；
- BF16 FFN input、gate、up、activated、down；
- block output。

每个边界报告完整Tensor的Max/RMS/relative-L2，以及B2两行是否相同。目标是找到第一个从FP32底噪
跃迁到BF16主误差的具体操作。只有该边界明确后，才设计一个最小精度或GEMM算法反驳实验。
