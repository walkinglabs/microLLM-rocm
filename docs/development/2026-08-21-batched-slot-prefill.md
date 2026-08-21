# Batched equal-length slot prefill

`forward_prefill_cached_rows()` now runs equal-length active prompts through one temporary batched
cache and maps each K/V prefix into arbitrary empty rows of a larger shared cache. The continuous
scheduler forms stable prompt-length groups at each admission boundary and records logical rows,
physical prefill batches and truly batched rows separately.

CPU/HIP tests cover partial row mappings, FP32/BF16, existing-row preservation, logits equality and
zero execution D2H. Alternating Release medians improve 2.931x at uniform R8/S8, 1.313x at R8/S4
and 1.056x at R8/S2. Uniform continuous reaches 87.4% of static batch throughput.

See [Experiment 101](../optimization-log/experiments/101-batched-slot-prefill.md) and the
[beginner guide](../dev/batched-slot-prefill.zh-CN.md).
