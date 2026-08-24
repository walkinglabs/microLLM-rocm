# Deferred HIP deallocation evidence

Experiment 176 adds an explicit, non-nestable scope that retains raw HIP allocations destroyed
by the current thread until one declared Stream completes. It does not route operators and does
not enable model Graph capture by itself.

## Files

- `matrix.jsonl` and `summary.json`: 8/32/128-node temporary chains × 1/4096 elements ×
  immediate-safe-sync/deferred × three fresh processes;
- `*-kernel-stats.csv` and `*-hip-api-stats.csv`: 32×4096 profiler control;
- `profile-summary.json`: extracted synchronization/API attribution;
- `coverage-summary.json` and `verification.json`: final regression evidence.

All 36 processes are exact and transfer-free. Deferred lifetime improves every row by
`2.28×–2.74×`. It deliberately trades temporary physical residency for fewer synchronizations:
the largest row retains 127 blocks / 2,080,768 bytes until scope completion.

Profiler keeps allocations, frees and Kernels unchanged, while Stream synchronizations fall
from 320 to 10 across ten repetitions.
