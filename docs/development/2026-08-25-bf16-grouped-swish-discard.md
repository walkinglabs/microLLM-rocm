# 2026-08-25 — 把一道工序塞进大机器，不代表全线更快

FFN原来先做两次矩阵乘法，再用SwiGLU合并两个结果。hipBLASLt能在gate矩阵乘法的
最后顺便做SiLU，但不能同时乘上up，所以还要一个小乘法kernel。

局部测试的稳定plan快约7%–10%，64个候选也都能算对。但整模中Qwen几乎不变，
DeepSeek反而慢约0.9%；小数值差经过很多层后也变成更大的logits差。

![Grouped Swish epilogue discard](../optimization-log/assets/bf16-grouped-swish-discard.svg)

因此这个能力留作显式实验，默认关闭。我们不再围绕这个小段调参。

最终回归为CPU 344/344、消毒342/342、PyTorch-enabled 318/318、完整CPU/HIP
542/542和HIP 186/186；只有3个既有条件跳过。
