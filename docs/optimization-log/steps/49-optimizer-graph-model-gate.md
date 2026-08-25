# Step 49 — Model optimizer Graph gate

Status: complete, reject model route

## Decision

21进程loss/参数精确、Graph两节点、optimizer metadata H2D归零，但Qwen T8/T512 optimizer为
0.798×/0.807×，DeepSeek T8为0.656×；完整step仅Qwen T8孤立1.050×，其余回退。模型路由
拒绝，optimizer-only Graph track关闭。
