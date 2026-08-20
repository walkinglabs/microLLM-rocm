# Step 09 — projection fusion and persistent operator plans

Status: `in progress` — bias epilogue discarded; Q/K bias+RoPE fusion kept

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

## Measured progress

- Experiment 011: hipBLASLt bias epilogue reduced allocations but regressed Qwen decode;
- Experiment 013: FP32 M=1 GroupedGemm returned no usable heuristic;
- Experiment 016: Q/K projection bias + split-half RoPE removed 1,120 launches in the
  DeepSeek trace and raised the fixed score `1.770568 → 1.784147`.

The remaining items are still separate hypotheses; the successful local fusion does not
prove that every neighboring pair should be fused.

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
