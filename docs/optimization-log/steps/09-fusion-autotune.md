# Step 09 — projection fusion and persistent operator plans

Status: `planned`

## Hypothesis

Once serial Kernels and allocator churn are removed, repeated small launches and
uncached hipBLASLt setup become the next bottleneck.

## Candidate experiments

Each bullet is a separate experiment:

- packed Q/K/V projection;
- packed gate/up projection;
- SwiGLU epilogue/fusion;
- residual add + RMSNorm;
- bias epilogue;
- cached hipBLASLt descriptors/layouts;
- heuristic algorithm selection;
- fixed workspace pool;
- offline exact-shape tuning;
- grouped GEMM.

## Plan key

```text
gfx + ROCm/hipBLASLt version + dtype
+ M/N/K + batch + strides + transpose
+ epilogue + alignment
→ algorithm + workspace + measured distribution
```

## Required tests

- plan cache invalidation by version/gfx;
- unsupported algorithm fallback;
- workspace ownership and Stream;
- exact same mathematical contract as unfused path;
- shape counterexamples retained.

## Keep gate

End-to-end improvement survives process restart, correctness passes, and the plan cache
does not silently use an algorithm for a neighboring shape.
