# 2026-08-25 — 能预分配，不等于应该增加 workspace API

公共API每次有3个逻辑分配，但都命中cache，没有新的backend allocation。

![BF16 weight-gradient workspace discard](../optimization-log/assets/bf16-weight-gradient-workspace-discard.svg)

Qwen preallocated Event略快，wall却只有0.986×；DeepSeek Event/wall是0.886×/0.889×。
这说明地址、plan或测量顺序的设备差异比host cache lookup更大，至少当前实现没有稳定wall收益。

所以不增加workspace类型。保留benchmark让未来backend复测，但当前track关闭。

