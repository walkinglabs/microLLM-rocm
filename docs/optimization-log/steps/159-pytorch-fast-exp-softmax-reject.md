# Step 159 — FP16 fast-exp rejection

Status: complete; candidate removed

近似exp通过10格精度/资源门，但width4096相对当前FP16 wave的Event/wall只有1.045×/1.034×，均
低于1.05。候选删除，保留精确`expf`。

详细记录见[Experiment 343](../experiments/343-pytorch-fast-exp-softmax-reject.md)。
