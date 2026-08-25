# Step 64 — direct BF16 P×V output

Status: complete, capability rejected

## Decision

interleaved BTHD和zero-stride GQA都返回hipBLASLt status 6，0个case进入计时，0个模型
路由。候选API全部撤回。
