# Step 06 — stream-aware caching allocator

Status: `complete` — Experiment 006, `keep`

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

## Kept stage

Only stage 1 plus the minimum Stream safety boundary was implemented:

- exact-size retired blocks;
- one disabled-timing Event records default-Stream completion;
- reuse only after Event readiness;
- pool disabled by default during load/warm-up;
- explicit enable for the measured steady-state interval;
- creating/passing any non-default Stream permanently disables reuse for that device;
- 8 GiB cached-byte cap and separately reported active/cached/reserved memory.

The first candidate enabled the pool too early and cached model-load temporaries,
doubling Qwen/DeepSeek inference reserved memory. It was rejected before the kept run.

## Measured result

```text
Qwen generation backend allocations       12,345 → 305
DeepSeek generation backend allocations   53,865 → 810
Qwen training backend allocations           9,200 → 1,154
DeepSeek training backend allocations      10,715 → 1,534
four-workload score                       1.219170 → 1.700597
```

Generation clears the original 10× gate; training reaches about 7–8×, an explicitly
recorded partial miss. Reserved memory remains near the pre-pool logical peak after the
enable timing was corrected.
