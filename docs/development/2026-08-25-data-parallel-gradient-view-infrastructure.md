# 2026-08-25 — gradient-as-bucket view infrastructure

## Problem

Persistent copy-backed buckets are faster but keep a second 124,689,408-byte unpacked-gradient
representation and still issue 114 D2D unpack copies per Model-S step.

## Change

- add an independent, default-off gradient-view policy;
- construct parameter-shaped contiguous Tensor views over each reduced bucket Storage;
- keep the exact parameter shape, dtype, device, offset, and optimizer-facing Tensor contract;
- remove all per-parameter unpacked Storage and unpack copies in view mode;
- reject views without persistent Storage and reject mode changes on an initialized plan;
- expose view counts and the policy through C++, CLI, recorded runs, and a three-policy runner.

## Pre-measurement evidence

The tiny two-rank gate proves every view in one bucket shares one Storage, offsets are exact prefix
sums, addresses survive the next backward/reduction, no unpack copies run, and two optimizer steps
keep rank parameters identical. The Model-S smoke records 114 views, zero unpacked Tensor Storage,
plan capacity 124,689,408 bytes, six first-step backend allocations, and zero later allocations.

The model route remains default-off until transient, persistent-copy, and persistent-view policies
run in rotated process order with exact loss/parameter and process-wide live/peak gates.

Infrastructure validation passes RCCL-labelled `30/30`; the machine audit registers 118 native
and Python test sources. The existing transient and persistent-copy controls remain independently
selectable in the same binary.
