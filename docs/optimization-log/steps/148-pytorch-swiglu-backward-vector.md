# Step 148 — SwiGLU backward vector4反例

Status: complete; candidate removed

六进程4K–16M矩阵证明vector/scalar只有0.946×–1.039×，0/2大shape过1.05；candidate API、
Kernel、test和runner清理。scalar producer本身已比可读native公式快2.07×–2.82×，所以下一步
转向zero-stride gradient materialization。

详细记录见[Experiment 332](../experiments/332-pytorch-swiglu-backward-vector-reject.md)。

