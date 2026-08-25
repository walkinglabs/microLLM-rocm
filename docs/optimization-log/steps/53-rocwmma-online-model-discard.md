# Step 53 — full-model online Attention gate

Status: complete, reject model route

## Decision

36个fresh processes准确命中168/196次native、零fallback；六格peak节省3.5–57MiB且top token相同，
但完整prefill只有0.761×–0.884×，Qwen最大logit Max/RMS为0.511/0.112。模型路由拒绝，公共
operator保留；只有消除RoPE后Q/K/V三次cast才允许重开。
