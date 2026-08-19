# 2026-08-19 — M4 zero-copy external TensorView operators

## Contract

Allow an external allocator such as PyTorch to retain data ownership while invoking
microLLM kernels. Inputs and output are non-owning TensorView descriptors; output
allocation, lifetime, Stream, and workspace remain caller responsibilities.

## API

```text
add_out(output, left, right, OpContext)
multiply_out(output, left, right, OpContext)
```

Version-one low-level operations require matching contiguous float32 shapes/devices.
They validate pointer, dtype, shape, stride, device, and Stream before execution.

## Verification

- a CPU test uses stack-owned arrays, proving no engine Storage is involved;
- a HIP test uses caller-created Tensor buffers and a non-default Stream, waits only
  that Stream, and obtains hand-calculated values;
- the HIP operation completes on MI300X in the dedicated conformance target.

This is the zero-copy operator seam required for PyTorch Custom Ops. Broader dtype,
stride, matmul workspace, and external C descriptor coverage remain later ABI work.
