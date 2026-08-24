# Caller-owned hipBLASLt Graph boundary

## Implemented

- `matmul_out_` for stable caller-owned output;
- CPU hand/transpose/error tests;
- HIP complete-output/address/zero-transfer tests;
- real hipBLASLt capture and two-replay conformance;
- Qwen/DeepSeek T512 repeated-GEMM matrix and profiler evidence;
- installed-package linkage of the new API.

## Result

Capture is supported by the current gfx942 runtime and each call becomes one Graph node. All 36
formal processes are bit-exact and transfer-free. The performance policy is rejected: Qwen reaches
0.906×/0.995×/1.022× and DeepSeek 0.902×/0.989×/0.990× at 1/8/32 calls.

Profiler shows launch compression but identical Kernel work. The caller-owned output API stays;
repeated vendor-only Graph routing does not. A future model region must mix the small Kernels that
benefited in Experiment 173 with GEMMs under one stable lifetime plan.

Full report: [Experiment 174](../optimization-log/experiments/174-hip-graph-gemm-discard.md).
