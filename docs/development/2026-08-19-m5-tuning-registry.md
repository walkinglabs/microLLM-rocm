# 2026-08-19 — M5 matmul tuning registry

## Contract

Let an offline benchmark or optimization skill override Auto selection for an exact
`(M,K,N)` shape without changing operator semantics. Reject abstract `Auto` entries,
invalid dimensions, and unavailable implementations.

## Selection order

1. validate HIP device, dtype, contiguity, rank, and inner dimensions;
2. build the exact dtype/layout/device/version/mode/workspace key and use a registered
   concrete implementation only when every field matches;
3. otherwise use the measured K/N-width heuristic;
4. fall back to readable for unsupported shapes/builds.

The registry is process-local and mutex-protected. The original shape-only key was later
replaced by `MatmulTuningKey`; see the
[exact-key follow-up](2026-08-23-matmul-registry-exact-key.md). A conformance test overrides
only FP32 NN 64 cube with hipBLASLt and proves FP16, TT, training-mode and different-workspace
lookups do not inherit it. Clearing restores readable selection.

## Boundary

This is the safe runtime registry seam, not yet a persistent autotuner. A future
offline tool can consume JSONL, run correctness gates, and register choices. Persistent
serialization and autotuning are still separate work. It must not select a candidate that
failed the CPU reference tolerance.
