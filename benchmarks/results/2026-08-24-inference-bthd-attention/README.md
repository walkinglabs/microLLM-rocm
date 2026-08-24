# Inference BTHD Attention island

Experiment 202 reuses existing layout-aware primitives to remove all four
per-block inference Attention materializations.

The explicit policy is eligible only for HIP, T≥256, BF16 Attention,
split-half RoPE with Q/K bias, no prefill-cache write, and no value-trace.
Every other case uses the previous BHTD path.

Twelve uninstrumented performance processes and twelve separate diagnostic
processes produce:

| Model | Baseline | BTHD | Speedup | Copy calls | Bytes removed | Peak saved |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 99,791 | 111,226 tok/s | 1.1146× | 96→0 | 100,663,296 | 4,194,304 |
| DeepSeek | 52,593 | 57,513 tok/s | 1.0936× | 112→0 | 205,520,896 | 7,340,032 |

All complete logits are bit-exact in the formal processes. Q/K projection
bias, split-half RoPE and BTHD→BHTD layout conversion run in one existing
operator. V stays BTHD, layout-aware causal GQA returns BTHD context, and the
output projection reshapes without copying.

Diagnostics are never enabled in throughput processes. The policy remains
explicit/default-off until more RoPE/bias/cache combinations are covered.

Files: performance-raw.jsonl, diagnostic-raw.jsonl, summary.json and
verification.json.
