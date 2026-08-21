# Experiment 053 evidence

- `raw.jsonl`: three Qwen T=512 matrix layouts × readable/hipBLASLt.
- `comparison.json`: valid speedups, invalid readable transpose timing, correctness scope
  and dispatch policy.

The transpose-left hipBLASLt result is correctness-valid, but its readable timing is not a
valid baseline because a temporary contiguous copy crossed streams. The record is retained
and excluded rather than silently repaired in the report.
