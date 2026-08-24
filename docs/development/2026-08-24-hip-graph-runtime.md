# HIP Graph runtime and readiness boundary

## What was added

- move-only `runtime::HipGraphExecutable`;
- explicit-Stream capture, node inspection, instantiation and replay;
- exception cleanup that restores the Stream after capture-unsafe allocation;
- CPU/undefined/move/device/lifetime-facing tests;
- a caller-owned fill/add benchmark and reproducible matrix runner;
- installed-package linkage of the new public runtime symbol.

## What was measured

On gfx942 MI300X, Graph loses at 1/8 tiny add nodes and wins at all 32/128/512-node rows.
Wall speedups range from 1.207× to 1.909× in the accepted region. All 60 fresh processes are
exact and transfer-free.

rocprofv3 keeps executed Kernel calls at 2,583 for the 128-node control but changes host
submission from 2,580 eager Kernel launches to 129 capture launches plus 20 graph launches.
Total traced HIP API calls fall 12,990→802.

## Honest boundary

This is runtime capability, not Qwen/DeepSeek acceleration. Model/autograd still allocate
temporary Storage and use implicit default-Stream operations. The runtime correctly rejects a
synchronous allocation inside capture and recovers for eager fallback. Model integration begins
only after a real vendor-GEMM caller-owned region and explicit Stream propagation pass.

Full report: [Experiment 173](../optimization-log/experiments/173-hip-graph-runtime.md).
