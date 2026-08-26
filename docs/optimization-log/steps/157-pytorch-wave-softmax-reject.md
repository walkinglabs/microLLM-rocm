# Step 157 — broad wave reduction rejection

Status: complete; candidate removed

width4096 wave reduction让FP16 Event/wall提高1.071×/1.070×，但BF16只有1.050×/1.033×，没有让
两个指标同时越过1.05。广义候选删除；cached shared-tree保持默认。

详细记录见[Experiment 341](../experiments/341-pytorch-wave-softmax-reject.md)。
