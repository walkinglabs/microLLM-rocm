# 2026-08-19 — M1 HIP runtime boundary

## Contract

Add HIP allocation, synchronous/asynchronous transfer, Stream, Event, device
inspection, and Tensor CPU/HIP transfer without adding operators or hiding global
device synchronization inside ordinary methods.

## Implementation

- `runtime::allocate/deallocate` own CPU and HIP allocation policy;
- `runtime::copy_bytes` is explicitly synchronous;
- `runtime::copy_bytes_async` requires a matching explicit Stream;
- Stream and Event hide HIP handle types from public headers;
- Event supports dependency insertion, readiness, synchronization, and elapsed time;
- Tensor transfer supports contiguous CPU/HIP tensors and rejects unsupported
  non-contiguous HIP materialization;
- cross-GPU copies are rejected until the distributed runtime defines semantics.

## Verification

CPU-only Debug build:

```text
21/21 tests passed
```

HIP Release build on the visible `gfx942` AMD Instinct device:

```text
22 tests discovered
21 passed
1 N0-only negative test skipped in a HIP build
```

The HIP tests queried device properties, transferred a Tensor CPU→HIP→CPU, used a
non-blocking Stream for H2D and D2H copies, recorded completion Events, and measured
non-negative device elapsed time.

## Review findings

The first HIP compilation found a handle-access typo in `Event::wait`; CPU-only
tests could not expose it. This justifies keeping both build modes as separate gates.

## Remaining M1 work

- readable CPU/HIP add and naive matmul;
- shared conformance cases;
- an explicit asynchronous-failure teaching example;
- HIP test labels separated from CPU labels.
