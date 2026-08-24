# Step 40 — Grouped weight-gradient 能力门

Status: complete, discard

## Decision

FP32 grouped `N,T` 与“共享转置 + `N,N`”在 Qwen/DeepSeek QKV/gate-up 共 8 case 中均为
0 supported candidate。Autograd 路由未创建。下一次只允许测试显式 packed 大 GEMM，并完整计入
pack/split成本。
