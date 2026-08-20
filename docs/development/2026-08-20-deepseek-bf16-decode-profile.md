# 2026-08-20 — isolated DeepSeek BF16 decode profile

The official HF CLI now isolates `prefill`, `decode`, or `both`. A decode-only rocprofv3
run for the pinned DeepSeek Distill model records 10,038 dispatches. GEMM accounts for
67.64% of Kernel time, fused cached Attention 10.22%, and BF16 casts 6.16%.

The exact call equation proves that four FP32 Attention Linear GEMMs per layer and the tied
FP32 output head remain outside the BF16 FFN island. Aggregate kernel/HIP API CSVs and a
machine-readable category summary are committed; the 18 MiB raw trace is reproducible but
not stored in Git.

No performance code changed in this node. It hands the next experiment a bounded target:
single-representation BF16 Attention projections while retaining FP32 cache/Norm math.
