# Width-selective GQA Value broadcast evidence

Experiment 170 routes only `D>=128 && repeats>1` through zero-stride Value P×V and dP.
Qwen D64 remains on the expanded route; DeepSeek D128 removes Value expansion in forward
and backward.

- `training.jsonl` / `summary.json`: disabled/enabled × Qwen/DeepSeek × three fresh T512
  processes;
- `baseline-kernel-stats.csv` / `broadcast-kernel-stats.csv`: complete DeepSeek profile;
- `profile-summary.json`: repeat and total-Kernel attribution;
- `coverage-summary.json`: post-change CPU coverage;
- `verification.json`: default-off decision and next forward-only boundary.

The complete route fails: Qwen is neutral-noisy at `0.9948×`; DeepSeek is `0.9972×` despite
112 fewer allocations. The profiler removes 168 Value-repeat calls but adds the same number
of KV-group GEMM dispatches; total Kernel time rises 0.67%. Engine and CLI defaults are false.
