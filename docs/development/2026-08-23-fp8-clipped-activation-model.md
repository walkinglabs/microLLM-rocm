# FP8 clipped activation model policy

`ModelConfig::fp8_activation_amax_fraction` connects the clipped dynamic operator to Tensor-amax
activations. The default is 1.0; valid experimental values are in `(0,1]`.

```text
--fp8-activation-scale-mode tensor-amax
--fp8-activation-amax-fraction 0.5
```

The fraction is included in model summaries, CLI JSON, benchmark rows and the experiment boundary.
Policy names gain a clipped-activation suffix when the value is below 1, so clipped and ordinary
Tensor-amax results cannot be silently grouped.

Only activation calls receive the fraction. Weight Tensor-amax and output-channel preparation still
use fraction 1.0. Shared QKV and gate/up activation quantization preserve their existing call count.

CPU and HIP tiny-model gates prove five dynamic activation calls and five clipped calls, value
stability across one-way weight preparation, finite outputs and zero hot-path payload transfer.
Official fraction search is still required; no clipped value is a default or a precision claim.

Exp151's valid coarse grid rejects 0.75/0.5/0.25: worst RMS is 6.55x--12.18x the 1.0 control.
Fractions 0.95/0.9/0.85 remain an explicit refinement gap. See
[Experiment 151](../optimization-log/experiments/151-fp8-clipped-coarse-grid.md).
