# FP8 Attention output-projection-only scope

Exp147's only no-regression failure is Qwen T512 RMS after scaling Q/K/V/O together. The new
`Fp8WeightScaleScope::AttentionOutputOnly` changes only each block's O projection. Q/K/V, all FFN
weights and the LM head remain device Tensor-amax.

```text
--fp8-weight-scale-mode output-channel-amax
--fp8-weight-scale-scope attention-output-only
```

This is a fresh scope alongside Attention-only, not a hidden change to its meaning. Policy names and
JSON distinguish both.

For a tiny model, exactly one weight has a column vector. Retained scale bytes are eight O columns
plus one scalar for every other Linear: 60 bytes untied and 56 tied. CPU preparation and inference
are value-stable. The HIP gate proves one column quantization, seven scalar quantizations, no
preparation D2H, no hot-path weight quantization and zero payload transfer.

The hypothesis is causal: if Q/K/V cause the long-context RMS regression, O-only should remove that
red bar while reducing post launches from four to one per layer. Same-revision device-Tensor control
must decide; this scope is not a default.

Exp148 meets the targeted keep rule: Qwen Max/RMS are unchanged, DeepSeek improves 7.75%--16.26%,
and both T512 regressions remain below 5%. Complete precision still fails 0/4, so the scope stays
opt-in. See [Experiment 148](../optimization-log/experiments/148-fp8-attention-output-only.md).
