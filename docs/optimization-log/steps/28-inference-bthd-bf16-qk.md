# Step 28 — direct BF16 Q/K into fused BTHD RoPE

Status: complete; explicit policy retained

## Evidence

- CPU/HIP算子覆盖FP32和BF16输入，并拒绝F16；
- grouped命中才能保留Q/K，普通与不匹配路径自动回到FP32；
- 两层整模测试证明retained dispatch计数与block数一致且logits位级一致；
- 五进程官方T512门为1.0224×/1.0238×，peak不变；
- phase profile删除48/56次cast，总Kernel改善1.0787×/1.0600×；
- 三进程DeepSeek只有1.0068×，作为稳定性反例保留。

## Decision

Keep the explicit/default-off policy. Expand sequence and batch before any default change.
