# 2026-08-19 — M3 complete Model-S training step

## Contract

Run one real training step through every Model-S layer and parameter. Allocate AdamW
first/second moments for the full 15.6M parameter set, compute cross entropy, execute
the complete readable backward graph, measure global gradient norm, and update a
parameter used by the input token.

## Observed result

```text
parameters=15586176
loss=11.2473
gradient_l2_norm=92.27
probe_parameter_before=-0.0178598408
probe_parameter_after=-0.0177598223
probe_parameter_delta=0.000100018457
wall_seconds≈0.96
```

CTest repeated the complete step successfully in 0.94 seconds on the current CPU
host. The time is a smoke observation, not a benchmark.

## Evidence boundary

This proves full Model-S CPU float32 training connectivity and optimizer state
allocation. It does not constitute a loss curve, real-corpus pretraining, validation,
peak-memory measurement, HIP backward, or a throughput result.
