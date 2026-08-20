# Experiment 035 retained-candidate profile

This repeats Experiment 033 after retaining BF16 Attention Q/K/V/O weights and shared QKV
input cast. The same pinned DeepSeek prompt generates the same 8 tokens.

The large raw trace is not committed. `kernel-stats.csv`, `hip-api-stats.csv` and
`profile-summary.json` are generated with `scripts/summarize_rocprof.py`.

```bash
rocprofv3 -f csv --kernel-trace --hip-runtime-trace -- \
  build/hip-release/apps/microllm_hf_infer ... \
  --bf16-ffn true --bf16-attention true --workload decode \
  --new-tokens 8 --warmup 0 --steps 1
```

Profiler wall time is not a throughput measurement. Experiment 034 and 036 contain the
un-instrumented repeated-process results.
