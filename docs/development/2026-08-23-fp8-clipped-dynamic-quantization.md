# FP8 clipped dynamic Tensor quantization

Dynamic Tensor-amax previously mapped the largest absolute E4M3 value exactly to 240. One extreme
element could therefore make the remaining Tensor use a coarse scale. `quantize_fp8_dynamic` now
accepts an optional `maximum_fraction` in `(0,1]` after the execution context.

```cpp
auto clipped = microllm::ops::quantize_fp8_dynamic(
    input, microllm::DType::Float8E4M3FNUZ, 1.0e-4F, {}, 0.5F);
```

The device scale becomes `amax * maximum_fraction / format_max`, bounded by the minimum scale.
Values outside the represented range are explicitly clamped before the FP8 cast. This avoids
depending on an implementation-defined overflow conversion.

`maximum_fraction=1` is the compatibility path. E4M3 uses 240; E5M2 now uses its own 57,344 finite
maximum rather than the old conservative E4M3 constant. A machine counter records clipped Tensor
calls separately from all dynamic calls.

CPU and HIP tests use a 0.5 fraction to prove that an input maximum of 100 reconstructs as the
clipped boundary 50. They also cover invalid fractions, E5 scale selection, device-resident scale,
and the unchanged default path. The CLI exposes the counter, but the Transformer has not yet opted
into clipping; model configuration and official scale search are separate nodes.
