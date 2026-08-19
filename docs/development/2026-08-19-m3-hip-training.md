# 2026-08-19 — M3 first complete HIP training trajectory

## Contract

Keep model parameters, forward activations, graph gradients, and AdamW moment tensors
on the HIP device while running a complete Transformer backward/update loop. Preserve
the CPU finite-difference path as oracle and state clearly where the readable GPU path
still uses host computation.

## Implementation

- Autograd Values now accept float32 CPU or HIP tensors;
- add/multiply/scale/matmul backward dispatch through HIP operators;
- transpose gradients materialize with the generic HIP stride-copy kernel;
- readable nonlinear backward formulas materialize values on the host, calculate the
  reference gradient, then transfer one Tensor back to the device;
- SGD and AdamW support HIP parameters and moment tensors, with readable host-side
  scalar update loops followed by device transfer;
- checkpoint snapshots GPU parameters/state through CPU serialization and restore to
  the parameter's original device;
- model input IDs and targets move explicitly to the model device.

## Observed MI300X trajectory

```text
step=1 loss=2.21512 gradient_l2_norm=28.7166
step=2 loss=2.33998 gradient_l2_norm=7.80848
step=3 loss=1.75952 gradient_l2_norm=9.11374
step=4 loss=1.46628 gradient_l2_norm=24.8076
step=5 loss=1.11681 gradient_l2_norm=5.45584
gpu=AMD Instinct MI300X VF
arch=gfx942:sramecc+:xnack-
```

Loss is not required to decrease every step; the acceptance gate is finite state and
final loss below first loss. The HIP CTest smoke completes in about 0.28 seconds.

## Performance boundary

This is a correctness-first AMD GPU training path, not a competitive performance
path. Nonlinear backward and AdamW currently cross the host boundary. M5 profiling
must make those transfers visible before device-native backward/update kernels replace
them. No tokens/s claim is attached to this result.
