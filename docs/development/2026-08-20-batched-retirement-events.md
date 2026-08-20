# 2026-08-20 — batched allocator retirement Events

The default-Stream exact-size allocator now retires sixteen blocks behind one shared HIP
Event. Shared ownership keeps the Event alive until every associated block leaves the
retired lists. Incomplete batches flush before explicit device synchronization, and
non-default Stream creation still disables reuse.

Measured MI300X evidence:

```text
Event record calls           8,993 → 562
fixed four-workload score     1.845199 → 2.470863
DeepSeek generation ratio     0.934716 → 1.251627
```

CPU `157/157`, sanitizer `155/155` and HIP `57/57` pass. The new focused test proves
sixteen blocks cross one completion boundary and all sixteen are reused without extra backend
allocations. Full evidence is in
`docs/optimization-log/experiments/022-batched-retirement-events.md`.
