# 2026-08-20 — Experiment 010 discarded gradient add_

A copy-on-write candidate used in-place FP32 accumulation only when gradient Storage had
one owner. Aliased leaves, repeated backward and full HIP Transformer gradients passed.

The primary measurement did not change:

```text
Qwen train logical allocations      9,200 → 9,200
DeepSeek train logical allocations 10,715 → 10,715
```

The source was removed. Future buffer reuse needs explicit graph contribution/liveness
information rather than a local Storage reference count.

Decision: `discard`.
