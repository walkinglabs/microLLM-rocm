# 2026-08-20 — batched allocator retirement Events

The default-Stream exact-size allocator now retires eight blocks behind one shared HIP
Event. Shared ownership keeps the Event alive until every associated block leaves the
retired lists. Incomplete batches flush before explicit device synchronization, and
non-default Stream creation still disables reuse.

Measured MI300X evidence:

```text
Event record calls           8,993 → 1,124
Event record API time        24.39 → 1.95 ms
fixed four-workload score     1.845199 → 2.389841
DeepSeek generation ratio     0.934716 → 1.205756
```

CPU `157/157`, sanitizer `155/155` and HIP `57/57` pass. The new focused test proves
eight blocks cross one completion boundary and all eight are reused without extra backend
allocations. Full evidence is in
`docs/optimization-log/experiments/022-batched-retirement-events.md`.
