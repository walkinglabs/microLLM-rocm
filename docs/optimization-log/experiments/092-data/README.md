# Experiment 092 data map

- `precision-summary.json`: bit-exact DeepSeek T2048 B1/B8 complete cached logits.
- `pair-raw.jsonl`, `pair-summary.json`: three alternating Release pairs for B1/B8.
- `gates.json`: focused tests, official precision, performance rejection and rollback.

One thread accumulates two independent Value columns, preserving each column's position order.
Numerics pass, but both performance medians regress about one percent. The candidate is not retained.
