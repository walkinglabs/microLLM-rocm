# E5M2 activation with E4M3 weight model policy

`Fp8ActivationFormat` exposes two FNUZ activation formats while Linear weights remain E4M3:

```text
--fp8-linear true
--fp8-activation-format e5m2-fnuz
```

E4M3 is the default. Non-default activation format is rejected for non-FP8 Linear models. Model
summaries, CLI JSON, policy identity and benchmark rows record the selected format.

Autograd now accepts independent left/right FP8 dtypes. E5 activation/E4 weight forward uses the
same mixed operand pair as graph-free inference; gradients remain FP32 straight-through updates to
FP32 masters. Existing calls default to E4/E4.

CPU gates cover mixed dequantized reference, FP32 gradients, full Transformer training and prepared
inference. HIP gates prove the mixed operator is native on MI300, model inference has no payload
transfer and output remains finite. These gates establish execution semantics, not official-model
accuracy.

The hypothesis is deliberately weak: E5M2 range is unnecessary after exact dynamic amax, while one
mantissa bit is lost. Official same-revision E4 control must determine whether model propagation
ever benefits from the format.
