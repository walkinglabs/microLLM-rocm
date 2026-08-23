# FP8 clipped activation model policy

The experiment added `ModelConfig::fp8_activation_amax_fraction` to connect clipped dynamic
quantization to Tensor-amax activations. The default was 1.0; experimental values were in `(0,1]`.

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

Exp152 closes that gap: 0.95/0.9/0.85 worsen worst RMS by 2.15x/4.98x/8.25x. The model/CLI policy
and pilot runner are removed after archival; the low-level operator remains. See
[Experiment 152](../optimization-log/experiments/152-fp8-clipped-fine-grid.md).

The removal is now complete. This file describes historical behavior; current `main` intentionally
has no model field or CLI fraction.
