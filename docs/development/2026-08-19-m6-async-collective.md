# 2026-08-19 — M6 asynchronous collective primitive

## Contract

Separate collective enqueue from waiting so communication can share a timeline with
independent compute. Preserve the existing synchronized average API as a composition
of asynchronous sum, communication-stream synchronization, and scale.

## API

- `enqueue_all_reduce_sum`: validate all ranks, group RCCL calls, enqueue on each
  rank's communication Stream, and return without stream synchronization;
- `synchronize`: wait rank communication Streams and abort the communicator if an
  asynchronous HIP/RCCL error becomes visible;
- `all_reduce(..., average=true)`: enqueue sum, wait, enqueue scale, wait.

A two-rank test enqueues sum for `[1,2]` and `[3,4]`, explicitly synchronizes, and
observes `[4,6]` on both GPUs. The next experiment will place independent compute on
separate Streams and compare serialized versus overlapped wall time.
