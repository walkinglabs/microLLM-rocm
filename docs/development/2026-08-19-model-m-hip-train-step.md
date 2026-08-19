# 2026-08-19 — 128MB-tier Model-M HIP training step

## Contract

Actually allocate and train the 31.3M-parameter Model-M on AMD GPU. Run full forward,
cross entropy, backward, AdamW update, and engine peak-memory tracking for one token.
Do not treat one step as a training curve.

## Observed MI300X result

```text
gpu=AMD Instinct MI300X VF
arch=gfx942:sramecc+:xnack-
parameters=31334912
fp32_weight_bytes=125339648
loss=9.134082794
gradient_l2_norm=111.950775146
probe_parameter_delta=0.000100007
peak_engine_hip_bytes=518798856
wall_seconds=3.189
```

CTest repeats the step in about 3.21 seconds. Peak bytes include microLLM-owned model,
gradient/activation, and AdamW state allocations observed during the step; they do not
include ROCm library/runtime external allocations.

## Evidence boundary

This fulfills the 128MB weight-tier execution and training-connectivity smoke. A real
Model-M training curve, validation loss, checkpoint artifact, mixed precision,
throughput benchmark, and dataset license/version remain unverified.
