# Ranked world-size boundary

Experiment 274 runs from clean revision `81fa7f0`.

| World size | Result | Rank diff | CPU max diff | Group time | Environment note |
|---:|---|---:|---:|---:|---|
| 1 | full pass | 0 | 1.4e-8 | 5.227 s | one rank/reference |
| 2 | full pass | 0 | 6.0e-8 | 5.328 s | existing two-GPU path |
| 4 | init failure | n/a | n/a | 2.756 s | 4/4 `ncclCommInitRank` system errors |

The machine exposes four MI300X virtual functions, but `/dev/shm` is only
67,108,864 bytes (64 MiB). Every four-rank process returns 1; none hangs and no
peer requires forced termination. This matches the previously observed RCCL
shared-memory boundary.

The worker and launcher now support a general `--world-size`: N rank-local
batches are combined into the CPU global reference, all rank identities and
parameters are checked, and process failure remains bounded. Interface support
does not prove that the current environment can initialize four ranks.

Reproduce with the three commands recorded in the subdirectory summaries. The
four-rank status may become a full pass on a host/container with sufficient shared
memory; until measured, this repository does not claim four-GPU execution.
