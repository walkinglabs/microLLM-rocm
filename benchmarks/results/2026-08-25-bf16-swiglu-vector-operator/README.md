# BF16 vectorized SwiGLU operator gate

The runner compares scalar and four-values-per-thread BF16 SwiGLU on the exact
B1T1024 Qwen and DeepSeek FFN activation sizes. Each row uses caller-provided
output Storage, three fresh processes, three warm-ups and 30 Event-timed calls.

Both complete outputs are bit-identical. Median mean-kernel speedups are 1.249x
for 4,980,736 elements and 1.190x for 9,175,040 elements, so the explicit
operator is admitted to the full-model gate.
