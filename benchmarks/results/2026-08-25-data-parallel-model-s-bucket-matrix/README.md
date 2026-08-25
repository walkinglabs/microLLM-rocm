# Model-S natural multi-bucket matrix

Experiment 254 runs Model-S B1T32 for five steps, with three rotated fresh
processes per 1/4/25 MiB policy. Step 1 remains visible; steady medians use
steps 2–5 and step 5 performs complete rank-parameter verification.

| Limit | Buckets | Communication | Total | Peak/rank | Relative to best |
|---|---:|---:|---:|---:|---:|
| 1 MiB | 45 | 9.18 ms | 21.76 ms | 549,089,280 B | 0.908× |
| 4 MiB | 12 | 15.49 ms | 28.29 ms | 549,089,280 B | 0.699× |
| 25 MiB | 3 | 6.825 ms | 19.76 ms | 603,383,808 B | 1.000× |

All 45 loss values match exactly; all nine final checks report zero rank
parameter difference. Every bucket plan covers 57 tensors and all 15,586,176
parameter elements.

The 25 MiB/3-bucket policy is the current reducer baseline. It saves 8.67 ms
versus 4 MiB but costs 54,294,528 additional peak bytes per rank. Bucket count
alone is not predictive: the 45-bucket path beats the 12-bucket path, so future
work must retain actual parameter ordering, pack/unpack and allocation evidence.

This is the first natural multi-bucket model workload. It enables a readiness/
overlap experiment; it does not yet claim overlap.

