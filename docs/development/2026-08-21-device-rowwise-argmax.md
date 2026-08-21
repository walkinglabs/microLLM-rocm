# 2026-08-21 — device row-wise argmax

Added CPU/HIP last-dimension row-wise argmax and routed uncached batched generation through
it. Shape, tie, non-finite, transfer and token tests pass. Same-card Qwen/DeepSeek B1–B8
gain 1.13x–2.15x with unchanged peak and tokens. Qwen B8 measured D2H bytes fall from
38,895,616 to 256 and throughput improves 2.19x.

See [Experiment 063](../optimization-log/experiments/063-device-rowwise-argmax.md) and
[raw evidence](../optimization-log/experiments/063-data/).
