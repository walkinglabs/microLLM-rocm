# Step 142 — Grouped gate/up cached-decode model gate

Status: planned

扩展已有显式CLI：cached decode时grouped key rows使用batch。Prefill rows4096不命中，64个rows2 decode
step命中每层一次。固定DeepSeek T2048/B2/N64、Arena baseline与Arena+65193 candidate、三轮交替进程；
要求64 token相同、吞吐≥1.01×、资源与dispatch计数通过。失败撤回decode扩展，成功也保持显式。
