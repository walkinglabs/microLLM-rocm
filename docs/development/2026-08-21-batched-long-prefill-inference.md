# 2026-08-21 — batched long-prefill inference

Graph-free model inference now reuses the public causal GQA operator instead of maintaining
a second composed Attention algorithm. T>=256 therefore uses retained strided-batched
hipBLASLt QK/PV; short sequences retain the fused readable path.

Focused MHA/GQA and graph-free model tests pass. Official Qwen/DeepSeek T512/T1024 prefill
improves 6.72x–16.73x; top tokens match, max top-logit difference is 0.195, and T128
improves 1.78x. T1024 adds 12%–33% peak and is documented as a speed/memory trade-off.

See the [experiment](../optimization-log/experiments/061-batched-long-prefill-inference.md)
and [raw evidence](../optimization-log/experiments/061-data/).
