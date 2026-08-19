# 2026-08-19 — M5 rocprofv3 runtime trace

## Contract

Use the installed rocprofv3 interface rather than assumed legacy flags. Capture HIP
API, kernels, copies, allocations, JSON, Perfetto, per-domain stats, and a summary for
the tiny HIP training failure.

## Command

```text
scripts/profile_hip.sh TRACE_DIR -- microllm_bench_model
  --mode train --model tiny --device hip
  --steps 2 --warmup 1 --batch 1 --context 8 --new-tokens 8
```

The run generated HIP API/kernel/allocation CSV traces, JSON, Perfetto, stats, and a
summary. The first script version passed an absolute summary filename and rocprofv3
nested it under its output naming scheme; using the relative name `summary` fixes the
path and is regression-tested by a repeat run.

## Evidence

- 756 `hipMemcpy` calls consumed 67.25% of traced HIP API duration;
- 792 `hipMalloc` and 792 `hipFree` calls occurred;
- 480 kernel launches occurred for three total steps including warm-up;
- device copy dispatches were 60.79% of kernel-domain duration;
- readable matmul kernels were 16.20%;
- strided copy kernels were 9.68%.

This strongly supports host transfer/allocation/launch overhead as the primary cause
of the tiny GPU slowdown. It contradicts “naive matmul is the only bottleneck.”
Optimizing only matmul cannot remove the 756 transfers or 792 allocation pairs.

Kernel and HIP API stats plus a manifest are committed. Full multi-megabyte trace and
Perfetto outputs are reproducible with the script and should be attached to a release
or experiment artifact store rather than every source commit.
