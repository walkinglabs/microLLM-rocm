# 2026-08-20 — immutable BF16 hipBLASLt plan cache

The new BF16 mixed/output GEMM path caches exact immutable descriptions and layouts per
thread. Diagnostic entries/hits/misses and clear APIs have CPU/HIP tests. The older rejected
general FP32 cache is not restored.

Three-process official medians improve Qwen decode/prefill `2.93×/2.74×` and DeepSeek
`2.55×/2.67×` versus the retained BF16 Attention path. Exact tokens/logit differences and
memory remain stable. Ratios versus full-BF16 PyTorch are all above one in the fixed matrix.

Full gates: CPU `164/164`, sanitizer `162/162`, HIP `64/64`, PyTorch oracle `4/4`.
