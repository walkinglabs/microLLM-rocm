# Step 54 — direct-BF16 model rebuttal

Status: complete, close online model track

## Decision

grouped QKV保留BF16 V、V bias和RoPE直接写BF16后，六格均略有恢复但仍只有0.777×–0.906×；
Qwen Max/RMS仍到0.485/0.110。三cast不是主要解释，模型路线再次拒绝并关闭track；通用BF16原语
与公共online operator保留。
