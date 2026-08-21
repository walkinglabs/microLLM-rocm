# KV Cache per-row position metadata

KV Cache now owns one logical position per batch row. Uniform callers retain `position()`;
divergent callers must use `row_position()` and an ambiguous uniform read throws. `advance_row()`
and `reset_row()` have strict row/capacity checks, while full reset restores uniform zero state.

The model intentionally does not consume divergent positions yet. This node establishes explicit
state and failure semantics before changing RoPE, K/V store and cached Attention kernels.

See [Experiment 084](../optimization-log/experiments/084-kv-cache-per-row-positions.md).
