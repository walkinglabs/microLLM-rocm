# 2026-08-19 — M1 explicit operator context

## Contract

Remove the hidden default-Stream assumption from the public operator layer. An
operator caller may supply a Stream and workspace without transferring ownership.
Default context remains available for simple code.

## Implementation

`OpContext` contains:

- an optional non-owning `runtime::Stream*`;
- an optional workspace pointer and byte count;
- validation that an explicit Stream matches the Tensor device.

All current HIP launches consume the context's native Stream. The context shape is
also the planned seam for benchmark workspace and PyTorch's current HIP stream.

The N1 executable creates CPU reference values, transfers inputs to the GPU, launches
add on a non-blocking Stream, records Events on that Stream, waits for completion,
and compares every element.

## Verification

```text
CPU-only regression: 30/30 passed
HIP conformance:       8/8 passed
N1 executable on gfx942:
  elements=4096
  maximum_absolute_error=0
  kernel_elapsed_ms=0.963837  (single observation, not a benchmark)
```

The elapsed value is retained only as proof that Event timing is connected. It is
not a performance conclusion: there was no warm-up or repetition.

## Negative case

Passing a CPU Stream to a HIP operator raises a device-mismatch error before launch.
This prevents a common integration bug from becoming an asynchronous memory error.
