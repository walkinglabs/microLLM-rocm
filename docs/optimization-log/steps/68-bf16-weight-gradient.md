# Step 68 — Cast-inclusive BF16 weight gradient

Status: in progress

当前 BF16 Linear 的 weight gradient 仍走 FP32 GEMM。先测
`cast+transpose(input) + cast(dY) + BF16 GEMM -> FP32 dW`，不先改 Autograd。

门：完整输出有限、BF16 CPU 抽样误差不超过 2e-3、每个 shape 三个新进程，operator
median 至少 1.05×且 minimum 至少 1.0×。小形状 32×64×96 的 0.823×反例永久保留。

