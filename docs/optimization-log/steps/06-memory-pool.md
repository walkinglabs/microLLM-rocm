# Step 06 — stream-aware caching allocator

Status: `planned`

## Hypothesis

Thousands of hipMalloc/hipFree calls and implicit synchronization materially increase
host/API time and prevent overlap.

## Staged design

1. exact-size reuse prototype;
2. bounded size classes;
3. per-device pool;
4. Event/stream-aware retirement;
5. reusable operator workspace arena;
6. optional hipMallocAsync pool backend.

Do not implement all stages in one experiment.

## Required tests

- zero-byte, alignment and large allocations;
- repeated acquire/release;
- two Streams with unfinished Event;
- exception/lifetime paths;
- current/peak/total counters;
- allocator shutdown after HIP runtime errors;
- no use-after-free under stress.

## Falsification

If allocation calls fall dramatically but measured tokens/s does not improve, the API
time was mostly waiting on earlier slow Kernels rather than allocator bookkeeping.

## Keep gate

- allocation/free calls reduced by an order of magnitude for fixed-shape steps;
- no global device synchronization;
- peak memory growth bounded and documented;
- all four workloads non-regressing.
