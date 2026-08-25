# Post-materialized DeepSeek T2048 cached-decode profile

This result profiles the retained no-flag default after Experiment 288. Both
fresh processes must report `cached_attention_materialized_policy=auto-enabled`;
otherwise the runner rejects the measurement.

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 \
  benchmarks/single_gpu/profile_current_cached_decode.py \
  --manifest /path/to/pinned-deepseek-manifest.json \
  --binary build/hip-release/apps/microllm_hf_infer \
  --output-directory \
    benchmarks/results/2026-08-25-post-materialized-deepseek-t2048-profile \
  --model deepseek-r1-distill-qwen-1.5b \
  --context 2048 --batch 2 --decode-tokens 64 \
  --warmup 1 --many-step-count 3 --overwrite
```

The runner subtracts the one-generation process from the three-generation
process and divides by two. Load and warm-up therefore cancel from the measured
generation. The derived workload contains 128 model forwards: batch 2 times 64
new tokens.

![Post-materialized kernel phases](profile-delta.svg)

Aggregate Kernel time is 831.31 ms per generation. Exact-order cached Attention
now consists of a 64.81 ms score phase and a 349.17 ms finalize phase. Together
they contribute 413.99 ms, or 49.80%; hipBLASLt GEMM contributes 272.79 ms, or
32.81%. Against the earlier same-workload profile, aggregate Kernel time is
1.2646x faster and application generation time is 1.2774x faster. The comparison
is historical rather than interleaved because the two profiles use different
binaries.

All 38,755 logical allocations in the derived generation are cache reuses and
backend allocation delta is zero. This rules out a steady-state workspace or
allocator change as the next primary experiment. The largest exact category is
the order-preserving finalize phase, so the next experiment changes only its
thread/work mapping and must preserve complete context and official-model logits.

`summary.json` and the raw CSV files are authoritative. `analysis.json` contains
the derived interpretation, `verification.json` pins the environment and gates,
and `profile-delta.svg` is a generated visual index.
