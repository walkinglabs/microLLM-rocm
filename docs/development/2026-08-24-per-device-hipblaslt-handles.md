# Per-device hipBLASLt handles

## Problem

Rank-local Transformer calls switched between GPU 0 and GPU 1, but hipBLASLt handles were static
singletons. The second device reused a handle created under the first and vendor GEMMs returned an
invalid-device error. Pure RCCL tests passed because their Streams/communicators already belonged
to explicit devices.

## Implementation

- added one thread-local hipBLASLt handle map keyed by device index;
- selected the device before lookup or handle construction;
- routed general, BF16, FP8 and interleaved Attention GEMMs through the same owner;
- retained BF16/Attention plan cache device keys;
- added direct alternating GPU FP32/BF16 coverage;
- added a 12-process single-GPU non-regression runner and contract test.

## Evidence

The RCCL multi-GPU suite is 11/11 and package consumers are 2/2. Previous failures reproduce on a
fresh detached `adcd642` build, while the new build passes. Qwen/DeepSeek T512 inference/training
remain exact and the minimum same-revision-boundary throughput ratio is 0.9979.

Full report: [Experiment 178](../optimization-log/experiments/178-per-device-hipblaslt-handles.md).
