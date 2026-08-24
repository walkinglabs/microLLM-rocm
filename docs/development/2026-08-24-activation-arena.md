# Stable HIP activation arena

## Implementation

- one stable HIP backing allocation bound to an explicit Stream;
- aligned, bounded monotonic slice planning and deterministic reset;
- Stream synchronization before backing destruction;
- two-slot liveness test and allocation-free Graph replay;
- deferred/eager/Graph 72-process matrix with setup break-even.

## Result

Eager arena improves all rows by 1.071×–1.768×. Arena Graph improves replay by
1.314×–3.066×, uses two stable addresses and contains only `N+1` Kernel nodes. Setup costs
14–16 ms, making break-even range from 9 to 1,280 replays.

The framework retains the explicit arena foundation. Model-wide routing is not enabled until a
real shape/liveness plan and complete model correctness gate exist.

Full report: [Experiment 180](../optimization-log/experiments/180-activation-arena.md).
