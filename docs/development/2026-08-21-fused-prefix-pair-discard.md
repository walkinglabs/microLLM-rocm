# 2026-08-21 — rejected fused prefix-pair cache store

Experiment 066 fused FP32-to-BF16 conversion and capacity-strided K/V prefix writes into one HIP
Kernel. The candidate passed CPU/HIP shape, dtype, stride, batch and zero-payload tests, removed every
measured D2D copy, and reduced a retained profile's Kernel time and dispatch count.

It was not merged. In the matched Release 72-process matrix, Qwen context2048 batch8 cache prepare
regressed from 400.372 to 522.553 ms and end-to-end generation from 543.785 to 658.594 ms. All three
candidate process runs reproduced the prepare regression. The implementation, public API and tests
were removed.

One independent contract hardening remains: paired step-store rejects mismatched current Key/Value
dtypes before HIP pointer dispatch. See
[Experiment 066](../optimization-log/experiments/066-fused-prefix-pair-discard.md) and its
[raw evidence](../optimization-log/experiments/066-data/).
