# 2026-08-20 — BF16-output GEMM foundation

`ops::bf16_matmul_output` accepts two BF16 matrices and explicitly selects FP32 or BF16
output while retaining FP32 accumulation. CPU rounds the final result for BF16 output;
MI300X uses a BF16 hipBLASLt D layout. Existing mixed `bf16_matmul` still returns FP32.

Focused CPU, HIP, zero-transfer and coverage tests pass. This is only the primitive
needed by a future FFN activation island; it is not a whole-model BF16 claim.
