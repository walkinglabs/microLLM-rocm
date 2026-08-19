# 2026-08-19 — M5 engine allocation tracker

## Contract

Measure memory owned by the microLLM allocator without claiming visibility into
driver, PyTorch, ROCm-library, or other-process allocations. Track CPU and HIP bytes
separately and preserve exact shared-Storage lifetime accounting.

## Metrics

- current live bytes;
- maximum live bytes since peak reset;
- total bytes allocated since reset.

Allocation is recorded only after successful host/HIP allocation. Deallocation
subtracts the exact Storage byte count after successful release. Peak reset starts at
the current live baseline rather than zero.

## Verification

A test captures the existing baseline, allocates 64 and 32 byte Storage objects,
observes current/peak increase by 96 and total allocation 96, then verifies current
returns to the baseline after both lifetimes end. It passes in CPU-only and HIP-enabled
builds.

This tracker will provide engine-owned peak memory in end-to-end benchmarks. ROCm
free/total memory remains separate diagnostic metadata.
