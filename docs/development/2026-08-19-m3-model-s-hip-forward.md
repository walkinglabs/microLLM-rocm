# 2026-08-19 — M3 complete Model-S readable HIP forward

## Contract

Transfer every Model-S parameter to one AMD GPU and run a real token through all six
layers using the readable HIP forward kernels. Compare every output logit with a CPU
model initialized from the same seed.

## Observed result

```text
gpu=AMD Instinct MI300X VF
arch=gfx942:sramecc+:xnack-
parameters=15586176
logits=8192
maximum_absolute_error=4.05312e-06
wall_seconds=0.933
```

The wall observation includes construction of CPU and HIP models, parameter transfer,
and both forwards. It is not a kernel or throughput benchmark. A separate tiny model
HIP test also matches the CPU cached forward within `2e-4`.

## Implementation boundary

`TransformerModel::to(Device)` explicitly moves all parameter tensors and clears old
gradients. The cached forward moves an input token to the model device. The first HIP
cache path materializes cache growth through host memory and currently supports the
one-token contiguous layouts exercised here. It proves numerical connection, not
efficient decoding or HIP training backward.
