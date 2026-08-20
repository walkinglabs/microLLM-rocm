# 2026-08-20 — prefill allocator boundary

The official HF CLI now supports independent `prefill`, `decode` and `both` workloads.
After prefill warm-up it enables the retained default-Stream exact-size allocator and resets
measurement counters before timing.

Three-process Qwen/DeepSeek prefill medians improve `1.64×/1.54×` for both FP32 and the
single-representation BF16 FFN policy. Decode stays within 1%, logits/tokens are unchanged,
and all CPU `161/161`, sanitizer `159/159`, and HIP `62/62` gates pass.

Against the fixed PyTorch full-BF16 reference, Qwen decode/prefill and DeepSeek prefill now
exceed 1.0. DeepSeek decode remains `0.522×` and is the next profiler target.
