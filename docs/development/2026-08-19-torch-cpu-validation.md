# 2026-08-19 — PyTorch Custom Op CPU validation

## Environment

An isolated temporary virtual environment installed the official PyTorch CPU wheel
`2.13.0+cpu`. It is not a project dependency and is not committed.

## Failures and fixes

1. `torch/extension.h` required Python development headers even though the library is
   loaded through `torch.ops.load_library`. Replacing it with `torch/library.h` and
   `ATen/ATen.h` removed the unnecessary Python C API dependency.
2. helper name `device_of` collided through argument-dependent lookup with
   `at::device_of`; it was renamed.
3. the Torch test depended on a Python interpreter variable local to a sibling CMake
   directory; the Torch directory now resolves its own interpreter.
4. importing `microllm.torch_ops` eagerly loaded the ctypes C API library. Package
   exports are now lazy, so the two optional bindings are independent.

## Evidence

`microllm_torch_ops` builds as a shared dispatcher library. `TorchOps.Basic` loads it,
calls `torch.ops.microllm.add/multiply` on CPU tensors, and matches PyTorch results.
The ROCm test is correctly skipped by a CPU-only Torch build.

PyTorch ROCm zero-copy/current-stream execution remains unverified and is still listed
as a release blocker for that specific integration claim.
