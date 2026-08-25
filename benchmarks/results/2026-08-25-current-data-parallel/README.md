# Current two-GPU data-parallel audit

Experiment 251 rebuilds the current RCCL configuration, passes 14/14 RCCL tests,
then runs 20 two-rank steps with a 4 MiB maximum bucket.

| Metric | Median | Share of total |
|---|---:|---:|
| Forward + backward | 1.565 ms | 68.34% |
| Communication | 0.350 ms | 15.28% |
| Optimizer | 0.070 ms | 3.06% |
| Unattributed parameter verification | 0.305 ms | 13.32% |
| Total | 2.290 ms | 100% |

Loss falls from 2.75 to 0.55 and maximum rank parameter difference is zero.
Step 1 contains 3.57 seconds of lazy setup and is visible in raw metrics; medians
keep it from becoming the steady result.

The tiny model fits in one bucket, so this run cannot demonstrate real backward/
communication overlap. The current implementation synchronizes every device after
complete backward, allocates and packs/unpacks buckets each step, and copies every
parameter to host for a rank-difference audit after every optimizer step.

The first production code contract is therefore measurement hygiene: preserve the
default every-step parameter check, time it separately, and allow an explicit sparse
interval for performance/production runs. Gradient-ready overlap remains a later
state-machine milestone.

Four-rank execution remains blocked in this container: `/dev/shm` is 64 MiB with
about 42 MiB available, below the already-recorded RCCL shared-memory requirement.

