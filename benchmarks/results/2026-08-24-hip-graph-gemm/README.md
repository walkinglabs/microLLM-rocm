# Caller-owned hipBLASLt Graph evidence

Experiment 174 adds `matmul_out_`: a caller-owned output boundary for readable/HIP and
hipBLASLt matmul. It then tests real Qwen/DeepSeek T512 GEMM shapes inside the retained
`HipGraphExecutable`.

## Files

- `matrix.jsonl` and `summary.json`: two shapes × 1/8/32 calls × eager/graph × three
  fresh processes with alternating order;
- `*-kernel-stats.csv` and `*-hip-api-stats.csv`: DeepSeek 32-call rocprofv3 control;
- `profile-summary.json`: extracted Kernel/API attribution;
- `coverage-summary.json` and `verification.json`: final regression evidence.

All 36 processes are bit-exact, output-address-stable and transfer-free. Captured node count
equals GEMM call count. Performance does not pass the two-shape gate:

| Shape | 1 call | 8 calls | 32 calls |
|---|---:|---:|---:|
| Qwen 512×896×896 | 0.906× | 0.995× | 1.022× |
| DeepSeek 512×1536×1536 | 0.902× | 0.989× | 0.990× |

The stable caller-owned output API remains useful infrastructure. Repeating an otherwise
independent vendor GEMM through Graph is rejected as a performance policy.
