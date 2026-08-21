# Experiment 101 data map

- `paired/uniform-r8s8/*.json`: three alternating pairs and one candidate profile row.
- `paired/r8s4/*.json`: divergent four-slot evidence.
- `paired/r8s2/*.json`: divergent two-slot evidence.
- `summary.json`: prefill grouping, performance and memory interpretation.
- `gates.json`: focused/full correctness counts.
- `environment.txt`: baseline, ordering and measurement contract.

Each profile row records logical `row_prefill_calls` separately from physical `prefill_batch_calls`,
so speedup cannot be produced by silently omitting requests.
