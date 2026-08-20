# 2026-08-20 — Experiment 007 discarded hipBLASLt plan cache

An exact-key, thread-local cache reused regular hipBLASLt operation descriptors and
matrix layouts. Focused NN/NT/TN/TT and FP32/FP16 tests passed, including hit/miss/reset
contracts.

The fixed end-to-end matrix rejected it:

```text
running best score    1.700597
candidate score       1.669755
Qwen generation          -6.1%
DeepSeek training         -5.2%
```

Candidate code was removed before documentation was committed. Raw JSONL is retained in
`docs/optimization-log/experiments/007-data/`.

Decision: `discard`.
