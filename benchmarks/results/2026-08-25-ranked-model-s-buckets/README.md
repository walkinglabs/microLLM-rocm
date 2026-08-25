# Ranked Model-S natural-bucket matrix

Experiment 266 compares per-parameter and 25 MiB synchronous bucket reduction on a
clean `c912202` revision. Each policy uses three fresh two-rank processes. Every rank
trains Model-S for one `B1×T32` step; a `B2×T32` CPU run is the global-batch reference.

| Policy | Collectives/rank | Reducer median | Reducer min–max | Training median | Group-wall median |
|---|---:|---:|---:|---:|---:|
| per parameter | 57 | 54.51 ms | 32.06–56.19 ms | 5657.56 ms | 9510.10 ms |
| 25 MiB bucket | 3 | 32.48 ms | 19.55–158.52 ms | 5648.32 ms | 9487.82 ms |

Bucketization reduces collective count 19×. Its reducer median is 1.678× faster, but
the three bucket samples have 89.3% coefficient of variation and include a 158.52 ms
cold-start spike. Complete training and group wall improve only 1.0016× and 1.0023×.
This is a measured correctness/baseline result, not a stable performance claim.

All 57 tensors and 15,586,176 values are compared. Rank/rank Max and RMS are zero;
rank/CPU Max is 0.0062738 and RMS is 3.483e-6. Mean local-rank loss differs from the
CPU global-batch loss by at most 9.555e-7. Peer failure remains bounded with return
codes `[1, -15]`. Temporary safetensors and communicator-ID files are not retained.

The next experiment records multiple steps inside each fresh process and separates the
first RCCL collective from steady steps. Persistent rank buckets or ready overlap are
not admitted from the noisy one-step reducer median.
