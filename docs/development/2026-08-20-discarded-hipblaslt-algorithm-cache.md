# 2026-08-20 — Experiment 008 discarded hipBLASLt algorithm cache

The candidate cached only the documented serializable `hipblasLtMatmulAlgo_t`, keyed by
shape, dtype, transpose flags and workspace. Focused cache and numerical tests passed.

A contemporary baseline contradicted the first historical comparison, so baseline and
candidate were each run in three independent processes. Per-workload medians showed:

```text
Qwen train        +0.3%
Qwen generation   -9.1%
DeepSeek train    -2.1%
DeepSeek generate -0.2%
median score      1.695566 → 1.646877
```

Candidate code was removed. The experiment established a prospective rule: claimed
gains below 10% require at least three baseline and three candidate process runs.

Decision: `discard`.
