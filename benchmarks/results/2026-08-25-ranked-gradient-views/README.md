# Ranked gradient-as-bucket view matrix

Experiment 269 compares four reducers from clean revision `490db1a`. Each policy
has three fresh two-rank groups; step 1 is cold and steps 2–3 produce six steady
samples per policy.

| Policy | Steady reducer | Steady step | Current/rank | Peak/rank | Unpack/step |
|---|---:|---:|---:|---:|---:|
| per parameter | 2.619 ms | 8.693 ms | 249.38 MB | 262.58 MB | 0 |
| transient bucket | 4.692 ms | 10.497 ms | 249.38 MB | 314.90 MB | 57 |
| persistent copy | 2.981 ms | 8.292 ms | 311.72 MB | 387.27 MB | 57 |
| bucket views | 2.662 ms | 8.242 ms | 249.38 MB | 324.93 MB | 0 |

Views improve reducer/step by 1.120×/1.006× versus persistent copies and
1.763×/1.274× versus transient buckets. Versus per-parameter, reducer is 0.984×
while complete step is 1.055×.

The view plan uses 62,344,704 bytes per rank, half of persistent-copy capacity.
It removes all 57 unpack copies and all later backend allocations. Final current
memory equals per-parameter; peak remains 62,344,704 bytes higher, but is
62,344,704 bytes below persistent copies.

All 15,586,176 values remain rank-exact after three steps. CPU Max/RMS are
0.0062715/3.701e-6, maximum mean-loss difference is 1.967e-5, and peer failure
remains bounded. Temporary weights and communicator IDs are removed.

Bucket views remain explicit and are not a default. They are admitted as the
storage prerequisite for a real one-process-per-GPU gradient-ready overlap test;
that experiment must preserve the same memory and correctness gates.
