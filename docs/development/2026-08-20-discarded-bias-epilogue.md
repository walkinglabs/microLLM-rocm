# 2026-08-20 — Experiment 011 discarded bias epilogue

hipBLASLt bias epilogue numerically matched CPU/PyTorch and reduced inference logical
allocations from 11,145→9,345 for Qwen and 48,545→40,565 for DeepSeek.

Three-process generation medians rejected it:

```text
Qwen       142.25 → 131.21 token/s  (-7.8%)
DeepSeek    53.04 → 54.13 token/s   (+2.1%)
score       1.752183 → 1.725932
```

Candidate code and API were removed. Fewer launches did not compensate for the slower
Qwen fused GEMM path.

Decision: `discard`.
