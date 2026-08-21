# Experiment 093 data map

- `summary.json`: position transitions, shared-Storage invariants and explicit serial boundary.
- `gates.json`: complete CPU/HIP/sanitizer counts and focused test dimensions.
- `environment.txt`: device/runtime and test commands.

This node is a correctness reference, not a performance result. `forward_cached_rows()` serializes
one B1 view per divergent row, so a future positions-aware HIP implementation can be accepted only
by matching these row logits, positions, Storage contents and error contracts.
