# Attention GEMM-alpha scale-fusion evidence

Experiment 167 replaces two explicit per-layer scale Tensors with hipBLASLt alpha:
QK forward uses alpha=`1/sqrt(D)`; dQ/dK each use the same alpha after unscaled softmax
backward.

- `training.jsonl` / `summary.json`: explicit/fused × Qwen/DeepSeek × three fresh T512
  processes;
- `*-diagnostics.json`: both policies preserve zero strided-copy state;
- `explicit-kernel-stats.csv` / `fused-kernel-stats.csv`: same-workload Qwen profile;
- `profile-summary.json`: scale-Kernel and total-Kernel attribution;
- `coverage-summary.json`: post-change CPU coverage;
- `verification.json`: final default-off decision and regression gates.

The candidate removes 96/112 allocations and saves 12.3 MB Qwen peak. The model gate is
mixed: Qwen regresses to `0.9869×`; DeepSeek reaches `1.0107×` but its fixed observed
parameter changes because scaling moved after FP32 accumulation. It therefore fails the
joint speed and parameter gates. The generic scaled-matmul primitive and explicit control
remain available; Attention fusion defaults false.
