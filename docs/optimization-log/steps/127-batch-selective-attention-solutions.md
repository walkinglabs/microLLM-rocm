# Step 127 — Batch-selective near-default Attention solutions

Status: planned

同一index策略失败的原因之一是B1/P×V 295716只有0.535×，而每个descriptor其实有自己的快候选。
下一反驳policy保持B1 default，并从Experiment 309选择：

- B2：QK default，P×V 295716（operator 1.003×）；
- B4：QK 311274（0.999×），P×V 295716（1.360×）；
- B8：QK 311303（1.096×），P×V 292462（1.038×）。

这些不同index不保证与B1位级相同，但各自相对default B1完整输出Max只有约`3e-7`/`4.5e-8`，远小于
原始跨batch漂移。CLI已经按exact descriptor注册，因此每个process只装当前batch的选择。

使用Experiment 310相同的完整cache/logits、两个fresh process、交错性能、peak/allocation门。若全局
Max/RMS不同时改善至少10%，或任一batch prefill<0.95×，拒绝并关闭Attention solution路线。
