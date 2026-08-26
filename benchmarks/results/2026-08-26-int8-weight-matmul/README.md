# INT8 weight matmul M=1 result

Three fresh processes each use two warm-ups and five measured iterations. Each process reports
its median; `summary.json` reports the median process. The C++ fused candidate is compared with
the C++ explicit-dequantize control, PyTorch per-call dequantize+matmul, and PyTorch resident-FP32
matmul. Complete outputs are checked before timing.

The fused path wins both per-call dequantize controls and removes the full floating-weight
temporary. It does not beat resident FP32 GEMM on DeepSeek, so `Auto` remains explicit-dequantize;
`FusedDecode` is retained as an explicit memory-first research implementation.
