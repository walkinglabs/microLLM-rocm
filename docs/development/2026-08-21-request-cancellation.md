# Request cancellation and immediate Cache release

## Problem

A serving scheduler cannot safely refill an empty slot until it can state exactly when the old
request stops owning memory. Completion already released KV Cache, but callers could not cancel a
request that was waiting or decoding.

## Contract added

- `ReferenceScheduler::cancel(id)` and `AdmissionBatchScheduler::cancel(id)` return `true` only for
  the first transition into `Cancelled`.
- `Completed` and `Cancelled` are terminal; cancelling either again returns `false`.
- an unknown ID raises `std::out_of_range` instead of silently doing nothing;
- a decoding request keeps its already generated token prefix but releases logits and KV Cache
  immediately;
- a cancelled admission request is excluded from compatibility grouping and GPU work;
- completed and cancelled counts remain separate.

## Evidence in this increment

- six CPU scheduler tests pass, including Cache bytes changing from non-zero to zero and a surviving
  request matching independent `generate()`;
- four HIP scheduler tests pass, including CPU/HIP survivor equality and cancelled-row exclusion
  from a two-row GPU batch;
- repeated cancellation does not change counters.

This is a lifecycle prerequisite for token-level slot refill. It is not itself a throughput
optimization and makes no speedup claim.
