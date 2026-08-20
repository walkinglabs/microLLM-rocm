# Experiment 038 training profile

`fp32/` and `bf16/` contain aggregate kernel/HIP API rows from one pinned Qwen train step.
The raw profiler traces are not committed. Reproduce each policy with:

```bash
rocprofv3 -f csv --kernel-trace --hip-runtime-trace -- \
  build/hip-release/apps/microllm_hf_train_step \
  --config /path/to/qwen/config.json --weights /path/to/qwen/model.safetensors \
  --tokens 1,2,3,4 --device hip --learning-rate 0.00001 \
  --warmup 0 --steps 1 --linear-precision bf16
```

Repeat with `fp32`, then aggregate with `scripts/summarize_rocprof.py`.
