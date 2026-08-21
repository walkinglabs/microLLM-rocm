# Experiment 088 data map

- `precision-summary.json`: complete cached-logit comparison for DeepSeek T2048 B1/B8.
- `gates.json`: small operator passes, official-model failures and rollback state.
- `environment.txt`: frozen baseline/candidate binaries and measurement boundary.

The full binary logits remain under `/tmp/microllm-exp088-precision/`; the committed summary records
their element counts, max error, RMSE and generated tokens. The candidate was stopped before timing
and fully removed from retained source.
