# MI300X official HF single-GPU matrix

This result contains official Qwen2.5-0.5B and
DeepSeek-R1-Distill-Qwen-1.5B inference and one-step training measurements.

```text
date          2026-08-20 UTC
device        AMD Instinct MI300X VF
architecture  gfx942:sramecc+:xnack-
compute       FP32
measurements  4/4 pass
```

The local manifest paths and checkpoint files are not committed. Model revisions,
expected parameter/Tensor counts, prompts and expected generated IDs are captured by
`benchmarks/single_gpu/hf_models.example.json`.

This is a short functional/performance smoke, not a stable serving benchmark. Memory is
engine-owned peak memory, not total board usage.
