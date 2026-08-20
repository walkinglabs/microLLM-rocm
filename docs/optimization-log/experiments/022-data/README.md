# Experiment 022 evidence

- `candidate-run-{1,2,3}.jsonl`: three complete official-model matrices, each with
  2 warm-ups and 5 measured iterations;
- `median-summary.json`: samples, medians, fixed-PyTorch ratios and score;
- `after-hip-api-stats.csv` / `after-kernel-stats.csv`: candidate DeepSeek profiler;
- baseline profiler: Experiment 018 `after-*-stats.csv`;
- baseline model medians: retained Experiment 018 row.

The large candidate PFTrace remains under `/tmp/microllm-exp022-profile/`.
