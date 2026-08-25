# Data-parallel parameter-verification interval matrix

Experiment 252 runs every-step, final-step-only and disabled parameter audits in
rotated fresh-process order. Each process has 20 steps; step 1 lazy setup remains
in raw evidence and steady medians use steps 2–20.

| Policy | Checks/process | Steady total | Verification | Speedup vs every-step |
|---|---:|---:|---:|---:|
| every-step | 20 | 2.96 ms | 0.395 ms | 1.000× |
| final-step-only | 1 | 2.38 ms | 0.370 ms when run | 1.244× |
| disabled | 0 | 2.52 ms | 0 | 1.175× |

All 180 loss values match exactly across policies. Final-step and disabled order
is tiny-workload noise; both demonstrate that every-step host verification is
outside the reducer hot path. Interval `1` remains the default correctness mode.

The implementation also fixes an implicit synchronization dependency: the old
host audit accidentally waited for asynchronous optimizer work. Optimizer
completion is now explicit and included in `optimizer_ms`, even when verification
is skipped.

