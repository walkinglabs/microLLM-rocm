# Ranked synchronous gradient-bucket matrix

Experiment 265 rotates per-parameter and one-bucket tiny reducers across three
fresh two-rank launches per policy. Every rank performs three steps and is checked
against the same CPU global-batch reference; the peer-failure test is repeated.

| Policy | Collectives/rank | Median rank-group time | Rank diff | CPU diff |
|---|---:|---:|---:|---:|
| per parameter | 36 | 5287.78 ms | 0 | 1.19e-7 |
| 4 KiB bucket | 3 | 5268.46 ms | 0 | 1.19e-7 |

Bucketization reduces collectives 12x but wall time improves only 1.0037x because
process, ROCm and RCCL startup dominates this tiny three-step workload. The route
is retained as a synchronous correctness baseline, not a performance result.

Peer failure remains bounded (`[1, -15]`). The next experiment uses a Model-S
one-step ranked workload to obtain natural multi-bucket evidence before persistent
Storage or ready-overlap migration.
