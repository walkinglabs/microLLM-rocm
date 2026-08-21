# 2026-08-21 — correctness-first serving scheduler

The inference library now includes a movable, non-copyable `ReferenceScheduler`. Requests may arrive
after prior steps, own independent B=1 KV caches and random generators, advance one token per
scheduler step, and release Cache Storage immediately at completion.

CPU and HIP tests compare delayed heterogeneous requests with independent `generate()` calls and
cover immediate completion, step limits, invalid prompts/policies and state/metric snapshots. A
106,816-parameter CPU/HIP benchmark records 1/2/4/8-request throughput, Cache bytes and sequential
equivalence.

The implementation intentionally contains no cross-request batched forward. It is the semantic
oracle for a future slot scheduler, not a production serving claim. See
[Experiment 072](../optimization-log/experiments/072-reference-serving-scheduler.md).

Final gates: full CPU/HIP 272/272, ASan/UBSan 187/187 and PyTorch-enabled CPU 192/192.
