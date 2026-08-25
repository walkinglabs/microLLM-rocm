# 2026-08-25 — gradient-ready Event overlap infrastructure

## State machine

1. step one builds persistent bucket views synchronously;
2. later steps reset per-rank/per-bucket parameter readiness;
3. the final parameter hook records one default-Stream Event;
4. when both ranks are ready, both communication Streams wait on their Event;
5. gradients pack into stable bucket Storage and RCCL sum plus in-place average enqueue;
6. after backward returns, one finish waits all communication and installs parameter views;
7. the optimizer starts only after communication and default-Stream work are complete.

Duplicate parameter readiness, incomplete buckets, changed parameters/devices, overlapping active
steps and unsupported config combinations fail explicitly. Ordinary training and the synchronous
view control are unchanged.

## Scope limitation

This is a single-process, sequential-rank controller. The second rank can overlap communication
with its own remaining backward work; rank0 does not continue computing. The experiment therefore
cannot establish standard DDP scaling and must hand off to one-process-per-GPU if retained.

## Pre-measurement evidence

Asynchronous average values pass, tiny three-step overlap matches the single-global-batch CPU
reference and exact rank parameters, and invalid config fails. Model-S pilot enqueues all three
buckets after warm-up, reports zero later communication allocations, roughly 1.48 ms finish wait,
14.01 ms steady total and unchanged view-path peak. Formal transient/sync-view/overlap-view A/B is
still required.

Infrastructure gates pass RCCL-labelled `36/36`; the machine audit registers 42 graph API entries
and 121 native/Python test sources.
