# Active-row compaction for continuous decode

`TransformerModel::forward_cached_active_rows()` advances a strictly increasing list of shared
cache rows and leaves every inactive row's full backing capacity unchanged. The continuous
scheduler now preserves the full-uniform batch fast path and otherwise gathers survivor tokens,
executes only active rows, and scatters compact logits back to fixed slots.

CPU/HIP tests cover FP32/BF16, independent-B1 logits, stable Storage, full inactive-capacity
preservation, zero HIP execution D2H and invalid row lists. Existing scheduler tests now require
zero dummy rows and exact inactive-row accounting.

Across the five MI300X Release divergent shapes, candidate throughput is 1.134x–1.348x the
Experiment 096 implementation and reaches 0.935x–0.985x the serial reference. Alternating A/B
medians retain 1.292x at R4/S4 and 1.226x at R8/S2 while reference drift stays below 0.8%.

See [Experiment 097](../optimization-log/experiments/097-active-row-compaction.md) and the
[beginner guide](../dev/active-row-compaction.zh-CN.md).
