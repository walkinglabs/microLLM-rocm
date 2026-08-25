# 2026-08-25 — 改完以后要重新看地图

旧的性能地图是在FFN Norm融合之前画的。新默认路径每层少一次cast，因此不能继续
拿旧图选下一个目标。

重新profile后，Qwen/DeepSeek的cast正好少24/28次，Kernel总时也下降。剩余的下一个小而
清楚的问题是：Attention前面的RMSNorm能不能也直接写入已有BF16 QKV工作区。

![Post FFN Norm profile](../optimization-log/assets/post-bf16-ffn-norm-profile.svg)
