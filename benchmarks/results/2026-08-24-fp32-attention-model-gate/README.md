# Exact FP32 Attention solution model gate

This directory records Experiment 189 on one gfx942 MI300X virtual function with
HIP runtime/driver `71399004` and hipBLASLt `1.3.0`.

The retained baseline is single-representation BF16 Attention/FFN inference plus
the accepted BF16 FFN Arena policy for flattened rows at least 512. The only changed
variable is which exact FP32 Attention GEMM receives a version-local solution index.

```bash
python3 benchmarks/single_gpu/compare_fp32_attention_solutions.py \
  --manifest /tmp/microllm-bf16-model-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-exp189-formal \
  --runs 3 --warmup 2 --steps 5
```

Each model runs `baseline`, `qk`, `pv`, and `both` in 3 fresh processes. Even-numbered
processes reverse the order. Qwen uses bit-exact QK/PV indices `311017/294519`;
DeepSeek uses `305423/292941`. The first QK recommendations from Experiment 188 were
not used because complete-model pilot logits exposed accumulated error.

| Model | QK | PV | Both | Max/RMS logits | Peak change |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.0093× | 1.0037× | 1.0082× | 0 / 0 | 0 B |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.9991× | 1.0029× | 1.0043× | 0 / 0 | 0 B |

All 24 processes are finite and bit-exact against the same-revision baseline. Every
candidate records the expected one or two registry entries, one first-use cache miss,
subsequent cache hits, and positive dispatches. Allocation calls and peak bytes are
unchanged. No policy reaches the required 1.01 speedup on both models, so no default is
enabled. The explicit registry and CLI controls remain available for future exact
environment experiments.

Files:

- `raw.jsonl`: 24 process records;
- `summary.json`: medians, complete-logit errors, memory and keep decision;
- `verification.json`: build and regression evidence for the accepted infrastructure.
