# Step 144 — Autograd外部梯度池完整模型门

Status: complete; explicit interoperability keep, default performance reject

先实现叶子级显式绑定与CPU/HIP生命周期测试，再让Tiny和Model-S的全部命名参数共享一块连续
显存。正式矩阵必须同时通过地址、完整Max/RMS和两个执行顺序，并测`zero + forward + backward`
的Event/wall、logical allocation与测量区峰值。

结果是一个有用反例：所有地址和梯度都完全正确，但三格Event中位数只有
0.871×/0.814×/0.792×，Model-S峰值还增加6.75–10.69MiB。接口保留给需要稳定地址的外部
系统；默认模型route不建立。

详细记录见[Experiment 328](../experiments/328-external-gradient-pool-discard.md)。

