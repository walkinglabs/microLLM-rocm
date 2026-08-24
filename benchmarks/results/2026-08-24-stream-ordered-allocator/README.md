# HIP Stream ordered allocator evidence

Experiment 179 tests the explicit HIP Stream Ordered Memory Allocator on one gfx942 MI300X. The
ordinary Storage allocator is unchanged.

Three policies execute the same add chain:

- `deferred`: synchronous ordinary allocations, one lifetime-region synchronization/release;
- `async`: `hipMallocAsync`/`hipFreeAsync` eager calls on one Stream;
- `graph`: capture the async allocation/free and add nodes, then replay one Graph.

The matrix covers 8/32/128/512 nodes, 1/4096 elements, three fresh processes per policy and 20
timed repetitions: 72 processes total.

| Result | Eager async | Graph allocation nodes |
|---|---:|---:|
| Speed ratio versus deferred | 0.619×–0.709× | 0.036×–0.048× |
| Unique temporary addresses | 2 | exactly node count |
| Graph node count | — | `3N+1` |
| Async pool reserved high | 128 MiB | outside default pool counters |

Every output is exact. Eager async proves same-Stream address reuse but loses on API/dependency
overhead. Capture does not preserve that two-address reuse: every graph allocation node owns a
distinct allocation and replay is much slower.

The 128×4096 profile executes 2,971 Kernels in all modes. Deferred/async/Graph Kernel duration is
5.60/7.89/10.96 ms. Graph compresses host Kernel launches from 2,967 to 129 plus 23 Graph launches,
but allocation nodes dominate the device timeline.

Files:

- `raw.jsonl`: 72 fresh-process rows;
- `summary.json`: eight shape comparisons and decision;
- `profile-summary.json`: parsed API/Kernel attribution;
- `*-hip-api-stats.csv`, `*-kernel-stats.csv`: raw profiler tables;
- `verification.json`: same-revision regression matrix.
