# 2026-08-19 — M5 matmul tuning registry

## Contract

Let an offline benchmark or optimization skill override Auto selection for an exact
`(M,K,N)` shape without changing operator semantics. Reject abstract `Auto` entries,
invalid dimensions, and unavailable implementations.

## Selection order

1. validate HIP device, dtype, contiguity, rank, and inner dimensions;
2. use an exact registered concrete implementation when present;
3. otherwise use the measured K/N-width heuristic;
4. fall back to readable for unsupported shapes/builds.

The registry is process-local and mutex-protected. A conformance test overrides the
normally-readable 64 cube with hipBLASLt, observes the registered choice, clears the
registry, and observes readable selection again.

## Boundary

This is the safe runtime registry seam, not yet a persistent autotuner. A future
offline tool can consume JSONL, run correctness gates, and register or serialize
choices keyed additionally by gfx architecture, dtype, and ROCm version. It must not
select a candidate that failed the CPU reference tolerance.
