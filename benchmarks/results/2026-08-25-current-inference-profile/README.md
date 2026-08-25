# Current default B1T1024 inference profile

This directory profiles the retained, default-off-for-online Qwen/DeepSeek BTHD
prefill path after the rocWMMA model track was closed.

```bash
ROCR_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/profile_current_inference.py \
  --manifest /path/to/pinned-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory benchmarks/results/2026-08-25-current-inference-profile
```

For each pinned model, rocprofv3 records load+1-prefill and load+6-prefill
Kernel stats. `(six - one) / 5` removes load, plan setup and lazy initialization.
The four application records prove BTHD and retained BF16 Q/K are enabled while
online Attention has zero native/fallback calls.

At B1T1024, hipBLASLt GEMM is 59.7% of Qwen Kernel time and 66.8% of DeepSeek.
Causal softmax is the largest named non-GEMM kernel at 14.8%/9.2%, but the prior
128-thread experiment offers only about 1% local T1024 gain. The next open,
bounded target is exact T1024 QK/PV solution screening rather than reopening the
closed softmax or online tracks.
