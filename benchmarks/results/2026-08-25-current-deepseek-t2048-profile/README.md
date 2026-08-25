# Current DeepSeek T2048 cached-decode profile

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/profile_current_cached_decode.py \
  --manifest /path/to/pinned-deepseek-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-current-deepseek-t2048-profile \
  --model deepseek-r1-distill-qwen-1.5b \
  --context 2048 --batch 2 --decode-tokens 64 \
  --warmup 1 --many-step-count 3 --overwrite
```

The runner subtracts the one-generation process from the three-generation
process and divides by two. Both have the same load and warm-up contract.

The derived generation has 1.051 seconds of aggregate Kernel time. Cached
Attention contributes 647.3 ms or 61.57%; hipBLASLt GEMM contributes 270.4 ms
or 25.72%. There are exactly 1,792 cached-Attention calls: 28 layers × 64
decoded tokens.

No backend allocation is added per generation. The 36,963 logical allocation
requests are all exact-size cache reuse. This rules out reopening the prior
allocator phase track for this workload.

HIP API durations are retained as raw evidence. In particular, synchronous
`hipMemcpy` duration can include earlier GPU work and is not reported as pure
copy time.
