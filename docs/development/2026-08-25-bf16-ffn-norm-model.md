# 2026-08-25 — 这次小零件真的让整辆车变快

新RMSNorm算子不再创建一张马上要丢掉的FP32表，而是直接写进FFN已有的BF16
工作区。它只在这个工作区确实存在时使用，其他路径不变。

两个真实模型的完整词表分数每一位都一样，整体快1.22%和0.92%。每个测量轮次还刚好
每层少一次内存申请。

![BF16 FFN Norm model gate](../optimization-log/assets/bf16-ffn-norm-model.svg)

现在开启BF16 FFN Arena就会默认使用它，但仍可用显式`false`回到旧路径检查。

最终回归为CPU 345/345、消毒343/343、PyTorch-enabled 319/319、完整CPU/HIP
544/544和HIP 187/187；只有3个既有条件跳过。
