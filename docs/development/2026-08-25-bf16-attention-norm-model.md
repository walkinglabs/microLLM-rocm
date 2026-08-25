# 2026-08-25 — Attention前面也不再写临时FP32表

上一步让FFN前的RMSNorm直接写BF16工作区。这一步用同样的规则处理Attention前的
RMSNorm，但只在QKV工作区确实命中时生效。

两个模型都快约1.3%，完整词表分数每一位一样，内存申请数和峰值同时下降。

![BF16 Attention Norm model gate](../optimization-log/assets/bf16-attention-norm-model.svg)

开启BF16 QKV Arena现在会默认开启该路径，也可显式关闭做对照。

最终回归为CPU 345/345、消毒343/343、PyTorch-enabled 319/319、完整CPU/HIP
544/544和HIP 187/187；只有3个既有条件跳过。
