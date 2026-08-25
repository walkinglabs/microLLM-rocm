# Step 51 — online rocWMMA causal GQA prototype

Status: complete, admit operator integration only

## Decision

42个fresh processes覆盖Qwen H14/KV2/D64与DeepSeek H12/KV2/D128、T32–2048。完整输出
全过，online/current为1.260×–4.041×，并在T2048删除224/192MiB全局score。短标量fused仍有
反例，batch/tail未实现，因此只准入带显式fallback的公共operator；模型路由关闭。
