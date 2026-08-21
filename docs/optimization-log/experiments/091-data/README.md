# Experiment 091 data map

- `precision-summary.json`: bit-exact DeepSeek T2048 B1/B8 complete cached logits.
- `pair-raw.jsonl`: three alternating baseline/candidate Release pairs for B1/B8.
- `pair-summary.json`: throughput medians, allocator, D2H, peak and token gates.
- `gates.json`: focused pass, official precision pass, performance rejection and rollback.

The candidate is numerically clean but adds a shared normalization pass and barrier without a
throughput benefit. It is retained as a performance counterexample, not retained source.
