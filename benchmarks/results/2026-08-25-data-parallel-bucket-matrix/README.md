# Tiny-model real bucket-count matrix

Experiment 253 scans four maximum bucket sizes with final-step parameter
verification. Each policy has three rotated fresh processes and 20 steps;
steady medians use steps 2–20.

| Limit | Buckets | Communication | Total | Relative to 4 MiB |
|---|---:|---:|---:|---:|
| 4 B | 12 | 1.26 ms | 2.98 ms | 0.725× |
| 64 B | 12 | 1.18 ms | 2.75 ms | 0.785× |
| 4 KiB | 1 | 0.34 ms | 2.03 ms | 1.064× |
| 4 MiB | 1 | 0.39 ms | 2.16 ms | 1.000× |

All 240 loss values match exactly and each process verifies identical rank
parameters on step 20. The 4 KiB and 4 MiB policies create the same single
bucket and therefore the same execution graph; their timing difference is
process noise, not a bucket crossover.

Twelve tiny buckets increase communication about 3–4x and slow total time. This
workload cannot demonstrate useful overlap: one bucket has nothing to overlap in
stages, while artificially tiny buckets are dominated by collective/pack overhead.
The next milestone must add a Model-S multi-bucket workload before any readiness
state machine is accepted.

