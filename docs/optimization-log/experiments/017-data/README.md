# Experiment 017 evidence

- `candidate-run-{1,2,3}.jsonl`: three inference-only candidate processes, each with
  2 warm-ups and 5 measured iterations.
- `after-kernel-stats.csv` / `after-hip-api-stats.csv`: candidate DeepSeek profiler.
- `median-summary.json`: candidate medians, retained Experiment 016 baseline medians and
  the fixed PyTorch ratios.

The baseline is exactly the three Experiment 016 candidate processes already committed
in [`016-data`](../016-data/README.md); they are not duplicated here. Training is
byte-for-byte the retained Experiment 016 path and its medians are reused.

The matching before profiler tables are `016-data/after-*-stats.csv`. Large candidate
PFTrace remains under `/tmp/microllm-exp017-candidate-profile/`.
