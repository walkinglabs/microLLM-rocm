# Official B2 prefill local-row audit

The official continuous CLI now accepts explicit per-request seed-token offsets. This makes row
order and duplicate-prompt experiments reproducible without changing model weights or scheduler
semantics.

Twelve DeepSeek processes compare P5 alone, `[P4,P5]`, `[P5,P4]`, and `[P5,P5]`. P5 has identical
B2 prefill and generated-token diagnostics whether it occupies local row zero or one. Both duplicate
rows are numerically identical, and all B2 full outputs match each other. B1 remains different.

The result refutes a local-row, stride, or cache-copy explanation. It supports a B1/B2 compute-shape
effect, but full-logit and per-block error growth remain the next numerical gates.

See the [beginner guide](../dev/prefill-row-audit.zh-CN.md) and
[Experiment 105](../optimization-log/experiments/105-b2-prefill-row-audit.md).
