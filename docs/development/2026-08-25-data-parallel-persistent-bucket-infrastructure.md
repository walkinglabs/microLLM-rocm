# 2026-08-25 — persistent gradient-bucket infrastructure

## Problem

The retained Model-S reducer still creates 6 bucket Tensor Storage objects and 114 unpacked
gradient Storage objects every step. RCCL communication streams cannot reuse the default-Stream
pool, so all 120 calls reach the backend.

## Change

- add a move-only `GradientBucketPlan` owned by `DataParallelTrainer`;
- bind it to communicator devices, parameter identity/order, shapes, and the bucket limit;
- allocate rank-local bucket and unpacked-gradient Storage after the first real backward;
- reuse the exact Storage addresses on later steps;
- reject persistent storage with allocating average, because replacement would break address
  stability;
- report temporary bytes and persistent plan capacity separately;
- expose the default-off C++/CLI/recorded-run controls;
- add a same-binary Model-S A/B runner with alternating process order.

## Correctness gates before measurement

The RCCL tests require move-only ownership, address stability, exact averaged gradients, explicit
contract-change rejection, zero communication-stage allocations after step one, identical rank
parameters, and safe clear/move behavior. The model route remains default-off until the formal
three-run Model-S matrix measures communication, total time, live bytes, and peak bytes.

Current infrastructure gates pass: CPU `357/357`, RCCL-labelled `26/26`, and the test-file audit
reports 117 registered native/Python sources. The direct Model-S smoke records 120 communication
allocations on step one and zero allocation/backend/cache calls on steps two and three.
