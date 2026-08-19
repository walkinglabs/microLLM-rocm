# 2026-08-19 — M3 real Model-S forward smoke

## Contract

Construct the complete configured Model-S rather than validating only arithmetic.
Run one CPU float32 token through six Transformer layers and the 8192-way vocabulary
head. Verify actual allocated parameter count and finite logits.

## Observed result

```text
parameters=15586176
fp32_weight_bytes=62344704
logits=8192
logit_min=-3.92813
logit_max=3.99042
wall_seconds=0.333
```

CTest repeated the executable successfully in 0.36 seconds on the current host. The
wall value is a single smoke observation with no warm-up and is not a benchmark.

## Evidence boundary

This proves that the 64MB-tier architecture can be allocated and executed end to end
through the CPU reference. It does not prove Model-S training, validation quality,
HIP model execution, peak training memory, or throughput.
