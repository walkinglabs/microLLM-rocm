# Experiment 016 evidence

| File | Meaning |
|---|---|
| `baseline-run-{1,2,3}.jsonl` | three independent processes built from retained commit `914b2d6` |
| `candidate-run-{1,2,3}.jsonl` | three independent fused bias+RoPE processes |
| `median-summary.json` | per-workload samples, paired medians and fixed-PyTorch ratios |
| `before-kernel-stats.csv` / `after-kernel-stats.csv` | DeepSeek 8-token rocprof Kernel tables |
| `before-hip-api-stats.csv` / `after-hip-api-stats.csv` | matching HIP API tables |

Every official-model process uses 2 full warm-ups and 5 measured iterations. Baseline
and candidate were run in the same session on the same MI300X VF. The fixed PyTorch
reference is unchanged.

The large PFTrace files remain in `/tmp/microllm-exp016-deepseek-profile/` and
`/tmp/microllm-exp016-candidate-profile/`; the compact source tables needed for the
reported counts are committed here.
