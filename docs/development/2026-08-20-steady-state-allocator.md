# 2026-08-20 — Experiment 006 steady-state HIP allocator

## Outcome

The engine now counts logical/backend allocations separately and can opt into an
exact-size, Event-retired HIP cache after warm-up. Any non-default Stream permanently
disables reuse and restores ordinary `hipMalloc/hipFree` behavior.

The first pool candidate was rejected because enabling during load doubled inference
reserved memory. The kept candidate enables only for the measured steady-state interval.

```text
CPU debug                    152/152 pass
ASan/UBSan                   150/150 pass
HIP release                   53/53 pass
PyTorch operator parity         4/4 pass
Qwen generate             93.34 → 134.87 token/s
DeepSeek generate         38.99 → 48.93 token/s
Qwen train                72.33 → 107.08 token/s
DeepSeek train            49.47 → 69.77 token/s
score                     1.219170 → 1.700597
```

Training backend allocations fell 7–8× rather than the planned 10×. Instrumented decode
also regressed because rocprof magnifies per-retirement Event API overhead. Both limits
are retained in the experiment report.

Decision: `keep`.
