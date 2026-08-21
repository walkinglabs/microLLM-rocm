# Experiment 074 evidence

- `raw.jsonl`: CPU/HIP × 1/2/4/8/16 requests × three fresh processes.
- `summary.json`: grouped throughput, reference speedup, group counts and maximum batch size.
- `build-contract.json`: compatibility pattern with groups of at most four requests.

The benchmark intentionally plateaus after B4 groups. It proves stable admission grouping, not
token-level continuous slot refill.
