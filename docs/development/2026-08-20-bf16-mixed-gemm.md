# 2026-08-20 — Experiment 014 BF16 mixed-GEMM foundation

The engine now has device-native FP32/FP16/BF16 cast and a BF16-input, FP32-compute,
FP32-output hipBLASLt path. It performs no payload host transfer.

```text
CPU debug       153/153 pass
ASan/UBSan      151/151 pass
HIP release      55/55 pass
PyTorch oracle     4/4 pass
shape speedups  0.83× to 1.15× FP32
```

Because three of five M=1 shapes regress, this is an operator foundation—not a claim
that the model supports or benefits from whole-network BF16.

Decision: `keep` foundation; model policy remains next work.
