# Step 27 — post-BTHD profile

Status: complete

## Evidence

- four trace processes and ten derived forwards;
- total Kernel speedups 1.169×/1.118×;
- strided category calls and time are zero;
- GEMM share 55.6%/65.2%;
- cast 0.519/0.757 ms; causal softmax 0.483/0.519 ms.

## Decision

Close measured BTHD layout work. Test BF16-input fused Q/K RoPE next.
