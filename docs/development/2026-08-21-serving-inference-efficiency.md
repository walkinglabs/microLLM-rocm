# Serving inference matrix and N64 pilot

The official-model matrix now has a named `serving` suite spanning contexts 1–2048, batches
1/2/4/8 and output lengths 1/8/32/64. Summary rows expose allocated-versus-active KV waste,
per-request waste, active-KV share of incremental peak and non-KV incremental bytes.

An independent MI300X pilot added Qwen/DeepSeek T1/32/128, B2/4, N64 evidence and a T2048/B2/N64
paired recheck. The long Qwen row is 1.250x PyTorch and the long DeepSeek row is 0.868x; both have
identical 64-token suffixes and exact BF16 KV accounting. One initial Qwen T128/B4 batch-row failure
did not reproduce in three fresh processes, so it remains preserved but is not classified as a
stable bug.

See [Experiment 095](../optimization-log/experiments/095-serving-inference-efficiency.md) and the
[beginner inference-matrix guide](../dev/inference-matrix.zh-CN.md).
