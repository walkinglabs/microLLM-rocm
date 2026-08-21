# 2026-08-21 — cooperative RMSNorm weight-gradient columns

For rows of at least 256, each RMSNorm weight-gradient column now uses one cooperative
HIP block instead of one thread serially scanning every row. Short-row behavior is unchanged.

Evidence: rows=256 across five model widths passes CPU/HIP forward and both-gradient
alignment; Qwen/DeepSeek T512 improve 1.220x/1.125x with unchanged peak; the target
Kernel improves 16.38x and all Kernel time improves 1.195x; T128 remains 1.003x.

See the [experiment report](../optimization-log/experiments/059-block-column-rmsnorm-weight-gradient.md)
and [raw evidence](../optimization-log/experiments/059-data/).
