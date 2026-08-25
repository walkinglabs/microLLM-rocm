# Current DeepSeek T2048/B2/N64 baseline

This directory is the clean-revision cross-framework baseline for Step 104.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /path/to/pinned-deepseek-manifest.json \
  --micro-binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /path/to/python-with-pytorch-rocm \
  --output-directory benchmarks/results/2026-08-25-current-deepseek-t2048 \
  --models deepseek-r1-distill-qwen-1.5b \
  --contexts 2048 --batches 2 --decode-lengths 64 --cases cached \
  --micro-kv-cache-dtype bf16 --micro-cache-capacity exact \
  --prefill-logits-mode last --warmup 2 --steps 5 --runs 3 \
  --timeout-seconds 900 --allow-amdsmi-fallback
```

The AMD SMI fallback is explicit. The underlying HIP runtime reports the device;
the first attempt omitted this flag and is recorded in `attempts.json` rather
than treated as a PyTorch measurement.

All 64 generated tokens match. Median throughput is 133.50 tok/s for microLLM
and 163.64 tok/s for PyTorch ROCm, a 0.8158x ratio. microLLM peak is 5.23 GB
versus PyTorch 6.38 GB. Both caches use exactly 121,110,528 bytes with 100%
utilization.

This is a performance failure baseline, not an optimization result. A current
rocprof trace is required before selecting the next kernel.
