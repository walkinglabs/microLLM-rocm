# Experiment 204 — profile after BTHD copy elimination

Status: measurement; BTHD layout track closed in measured domain

| Model | Pre-BTHD Kernel | BTHD Kernel | Speedup | Strided calls |
|---|---:|---:|---:|---:|
| Qwen | 5.680 ms | 4.858 ms | 1.169× | 96→0 |
| DeepSeek | 10.160 ms | 9.085 ms | 1.118× | 112→0 |

![Post-BTHD profile](../assets/inference-bthd-profile.svg)

GEMM remains 55.6%/65.2%. Cast takes 0.519/0.757 ms and causal softmax's top Kernel takes
0.483/0.519 ms.

The next minimal candidate is BF16-input fused Q/K bias+RoPE, which can remove two casts per block
without changing V or the Attention output contract. Cached-prefill and generic layout work remain
outside this hypothesis.

Raw evidence:
[benchmarks/results/2026-08-24-inference-bthd-profile/](../../../benchmarks/results/2026-08-24-inference-bthd-profile/).
