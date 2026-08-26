# Step 151 — torch.compile反例

Status: complete; rejected

八进程64K/1M矩阵中compiled/eager仅0.584×–0.610×，compiled/native 0.462×–0.476×，cold
55.8–1160.3ms；梯度过门、loss归约差单独记录。compiled推荐拒绝，最后相邻候选为C++ Autograd。

详细记录见[Experiment 335](../experiments/335-pytorch-swiglu-compile-reject.md)。

