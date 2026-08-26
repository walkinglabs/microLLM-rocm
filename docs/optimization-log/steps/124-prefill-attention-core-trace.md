# Step 124 — Full-prefill Attention core trace

Status: completed by Experiment 308

Experiment 306在exact Q/K/V和BF16 cache之后，将第一处差异定位到Block 0 Attention context。当前
context由QK、scale、causal softmax、P×V和layout组成。

固定DeepSeek T2048、Q=296100、K/V=292135、B1/B2/B4/B8、两个fresh process，记录前两个batch row：

- raw QK scores与scaled scores；
- causal softmax probabilities；
- P×V输出（layout前）；
- final context。

每个阶段报告完整Max/RMS/relative-L2与within-batch bitwise。由于T2048 score/probability很大，trace
只允许Block0和前两个row，并按单阶段分进程采集，避免一次保留全部Tensor。任何trace结果不用于性能。

完整二进制结果发现batch-dependent双机制：B2的scores/probabilities exact，P×V首差；B4/B8的
causal-visible scores首差。所有within-batch行exact，两个process完全重复。详见
[`Experiment 308`](../experiments/308-prefill-attention-core-matrix.md)。默认不变。
