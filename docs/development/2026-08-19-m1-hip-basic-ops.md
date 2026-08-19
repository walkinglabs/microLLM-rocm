# 2026-08-19 — M1 readable HIP basic operators

## Contract

Add direct HIP kernels for contiguous float32 fill, add, multiply, scale, and equal-
batch-rank matrix multiplication. Kernel launch remains asynchronous. The result is
accepted only through comparison with the CPU reference path.

## Implementation

- one-thread-per-element kernels for fill and elementwise operations;
- intentionally naive one-thread-per-output batched matmul;
- launch error checks without device-wide synchronization;
- explicit rejection of mixed devices and non-contiguous HIP input;
- a dedicated HIP conformance test target and `hip` CTest label.

The matmul is a teaching/reference GPU path, not a performance claim. Tiled and
hipBLASLt paths are later implementations behind the same semantics.

## Verification

```text
CPU-only build: 30/30 passed
HIP build:      33 passed, 1 N0-only test skipped
HIP conformance label: 3/3 passed on gfx942
```

## Failure found during implementation

Linking `hip::device` directly to a mixed CXX/HIP target propagated an AMD offload
compiler flag into ordinary `g++`, which rejected it. The fix was to let CMake's HIP
language compile only the `.hip` source; the runtime target already supplies the
required host runtime linkage.

This is retained as a build-system failure example: a kernel can compile correctly
while the mixed target still fails because compile options crossed a language
boundary.

## Remaining M1 work

- Embedding, Softmax, RMSNorm, SiLU/SwiGLU, RoPE, and cross entropy HIP paths;
- an N1 CPU/HIP artifact and asynchronous-read failure demonstration;
- explicit Stream-aware low-level operator launch API.
