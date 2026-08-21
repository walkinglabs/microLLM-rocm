# Experiment 094 data map

- `summary.json`: row-prefill state transitions and semantic contracts.
- `gates.json`: complete CPU/HIP/sanitizer counts and focused dimensions.
- `environment.txt`: device/runtime and focused test names.

This experiment is a correctness reference. It intentionally does not contain a speedup number:
the retained implementation creates a temporary B1 cache and copies K/V by layer and head.
