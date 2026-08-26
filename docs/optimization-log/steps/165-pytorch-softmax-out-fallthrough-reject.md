# Step 165 — Softmax-out Autograd fallthrough rejection

Status: complete; removed/line closed

fallthrough保持10格正确，但FP16/BF16 width4096仅约1.008×/0.998×当前显式Autograd kernel。候选删除，
adapter局部提交线关闭。

详细记录见[Experiment 349](../experiments/349-pytorch-softmax-out-fallthrough-reject.md)。
