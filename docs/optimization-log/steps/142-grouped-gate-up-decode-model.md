# Step 142 — Grouped gate/up cached-decode model gate

Status: completed by Experiment 326; rejected

扩展已有显式CLI：cached decode时grouped key rows使用batch。Prefill rows4096不命中，64个rows2 decode
step命中每层一次。固定DeepSeek T2048/B2/N64、Arena baseline与Arena+65193 candidate、三轮交替进程；
要求64 token相同、吞吐≥1.01×、资源与dispatch计数通过。失败撤回decode扩展，成功也保持显式。

结果：180.19/178.46 tok/s=1.00968×，低于1.01；tokens相同，候选拒绝并撤回decode扩展。
