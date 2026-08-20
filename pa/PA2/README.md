# PA2 — 从稳定失败提出下一版系统

Start from a failure measured in PA1 or the repository failure atlas. Do not begin
with a desired architecture component.

## Required structure

```text
Observed stable failure:
Minimal reproduction command and raw artifact:
Explanation A:
Explanation B:
Evidence currently favoring one explanation:
One minimal system change:
Experiment that could falsify the favored explanation:
Expected observation:
Actual observation:
What the result supports:
What the result does not support:
Next-version proposal:
```

## Eligible repository failures

- tiny HIP train/generation slower than CPU;
- low-loss cycle fails beyond trained context;
- hipBLASLt steady-state improvement with startup regression;
- four-rank RCCL initialization fails under 64MB `/dev/shm`;
- synthetic communication/compute overlap exists, but backward-ready overlap is not
  implemented.
- pre-quantized FP8 GEMM is faster on one shape, but official Qwen whole-model FP8
  speedup is still unverified;
- Qwen/DeepSeek cached generation is correct, but device-native preallocated cache and
  end-to-end decode profiling remain follow-up work.

## Agent boundary

An Agent may locate code, scaffold one experiment, parse logs, or generate a candidate
kernel. The learner owns the causal explanation, falsification experiment, evidence
scope, and merge/reject decision.
