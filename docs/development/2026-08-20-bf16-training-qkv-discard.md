# 2026-08-20 — BF16 training shared-QKV candidate discarded

A paired Qwen profile showed 360 added BF16 cast launches: GEMM saved 1.33 ms while cast
Kernels cost 1.91 ms. A shared-QKV autograd candidate removed exactly 240/280 allocations
for Qwen/DeepSeek over five steps and matched independent STE gradients.

Three-process throughput was `0.973×/1.009×` the BF16 baseline, geometric `0.991×`.
The graph API/model path/tests were removed; raw results and the inference-only shared-QKV
operator remain. BF16 training still needs a larger continuous island.
