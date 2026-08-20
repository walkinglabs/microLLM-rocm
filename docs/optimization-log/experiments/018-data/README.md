# Experiment 018 evidence

- `deepseek-run-{1,2,3}.jsonl`: three direct DeepSeek official-checkpoint processes,
  each with 2 warm-ups, 5 measured iterations and 8 generated tokens.
- `after-kernel-stats.csv` / `after-hip-api-stats.csv`: 512-thread candidate profiler.
- the 256-thread baseline is Experiment 017, including
  [`017-data/after-kernel-stats.csv`](../017-data/after-kernel-stats.csv).

Qwen uses width 896 and therefore executes the byte-identical 256-thread path. Both
training paths are unchanged. The large PFTrace remains in
`/tmp/microllm-exp018-profile/`.
