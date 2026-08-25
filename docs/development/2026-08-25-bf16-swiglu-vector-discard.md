# 2026-08-25 — 一个小零件快了，整辆车不一定快

SwiGLU像FFN中的一个门：两排数字进来，一排数字出去。新kernel让每个GPU线程
一次处理4个BF16数，两个真实shape的小测快1.25倍和1.19倍，答案每一位都相同。

但整个模型还要做许多矩阵乘法、Attention和数据转换。Qwen整体只快0.73%，DeepSeek只快
0.05%。DeepSeek没过事先写好的0.5%门。

![BF16 SwiGLU vector discard](../optimization-log/assets/bf16-swiglu-vector-discard.svg)

所以新算子留给后续实验显式使用，默认模型不改。下一次要动更大的一段，不能只打磨这个
小零件。

最终回归为CPU 344/344、消毒342/342、PyTorch-enabled 318/318、完整CPU/HIP
542/542和HIP 186/186；只有3个既有条件跳过。
