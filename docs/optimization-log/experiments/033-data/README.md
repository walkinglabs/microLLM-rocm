# Experiment 033 profiler evidence

This profile isolates `--workload decode` for the pinned DeepSeek Distill checkpoint with
12 prompt tokens and 8 generated tokens. It uses the Experiment 032 BF16 FFN policy.

```bash
rocprofv3 -f csv --kernel-trace --hip-runtime-trace -- \
  build/hip-release/apps/microllm_hf_infer \
  --config /path/to/deepseek/config.json \
  --weights /path/to/deepseek/model.safetensors \
  --tokens 151646,151644,3838,374,220,17,10,17,30,151645,151648,198 \
  --device hip --new-tokens 8 --warmup 0 --steps 1 \
  --bf16-ffn true --workload decode

python3 scripts/summarize_rocprof.py \
  --kernel-trace /path/to/trace_kernel_trace.csv \
  --hip-api-trace /path/to/trace_hip_api_trace.csv \
  --output-directory docs/optimization-log/experiments/033-data
```

The 4 MiB kernel trace and 14 MiB HIP API trace are not committed. The complete aggregate
rows and the category summary are committed. Profiler wall time is invalid because module
load and instrumentation dominate; use Experiment 032 for throughput.
