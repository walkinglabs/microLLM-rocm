# Post-BTHD T512 phase profile

Experiment 204 profiles the composed grouped policy after the inference BTHD
island is enabled.

| Model | Total Kernel | Speedup vs pre-BTHD | Strided calls | GEMM share |
|---|---:|---:|---:|---:|
| Qwen | 4.858 ms | 1.169× | 96→0 | 55.6% |
| DeepSeek | 9.085 ms | 1.118× | 112→0 | 65.2% |

The next non-GEMM categories are:

| Category | Qwen | DeepSeek |
|---|---:|---:|
| FP32/BF16 cast | 0.519 ms | 0.757 ms |
| causal softmax top Kernel | 0.483 ms | 0.519 ms |
| RMSNorm | 0.243 ms | 0.385 ms |

The BTHD layout track is closed for the measured no-cache domain. A distinct
next candidate is to let fused Q/K RoPE read grouped BF16 projection outputs
directly, removing two BF16→FP32 casts per block. V precision and cached
prefill remain separate questions.

Files: per-model one/six-step Kernel stats, profile-delta.json, summary.json
and verification.json.
