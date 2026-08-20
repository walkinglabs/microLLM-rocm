# 2026-08-20 — BF16 Attention shared Q/K/V cast

`prepare_bf16_attention_inference()` transactionally replaces Q/K/V/O FP32 projection
weights with one BF16 representation. `bf16_qkv_projection` casts their shared normalized
input once, submits three BF16-weight GEMMs, and returns FP32 values for the existing
bias/RoPE/cache path.

The first per-Linear-cast pilot regressed DeepSeek and is retained. The shared-cast version
passes CPU `163/163`, sanitizer `161/161`, HIP `63/63`, and Python/PyTorch oracle `4/4`.

Three-process Qwen decode/prefill improve `2.90%/6.89%`; DeepSeek changes
`+1.98%/-2.72%`. Exact tokens pass and persistent weights fall another 88 MB/308 MB.
DeepSeek decode remains `0.533×` PyTorch BF16, so broader BF16/output-head work is not yet
accepted.
