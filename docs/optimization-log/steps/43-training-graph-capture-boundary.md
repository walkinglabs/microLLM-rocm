# Step 43 — 完整训练HIP Graph边界

Status: complete, reject full-step Graph

## Decision

FP32/BF16各四阶段、三次新进程共24/24稳定。forward/backward/full-step因动态Tensor Storage被
安全拒绝；AdamW捕获21个设备节点，但Graph重放不推进主机step。保留分配保护和诊断工具，
完整训练Graph必须等待图级liveness plan与device-owned optimizer state。
