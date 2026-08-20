# 2026-08-20 — Python/PyTorch ROCm performance comparison

## Contract

The baseline must be Python PyTorch on the same MI300X, not LibTorch inside microLLM,
CPU timing presented as GPU timing, or a manually copied marketing number.

Built-in rows require exact equality of model, parameters, FP32 weight bytes, dtype,
batch, context, warm-up, repetitions and measured tokens. Official rows additionally
require the same checkpoint Tensor count and generated IDs.

## Environment

```text
PyTorch       2.11.0+rocm7.13.0rc2
Transformers  4.55.4
HIP           7.13.99004
GPU           AMD Instinct MI300X VF, gfx942
```

The wheel's AMDSMI path returned zero devices although its private HIP runtime count
returned four and a real CUDA-named PyTorch tensor ran on MI300X. The explicit
`amdsmi_zero_fallback_to_hip_runtime` workaround is present in every PyTorch row.

## Built-in comparison-grade result

| Model | Mode | microLLM token/s | PyTorch token/s | microLLM/PyTorch | Peak memory ratio |
|---|---|---:|---:|---:|---:|
| tiny | train | 3917.81 | 1730.36 | 2.264 | 0.001 |
| tiny | generate | 854.84 | 792.25 | 1.079 | 0.0004 |
| Model-S | train | 13.57 | 177.57 | 0.076 | 0.526 |
| Model-S | generate | 139.22 | 293.55 | 0.474 | 0.437 |
| Model-M | train | 3.51 | 59.94 | 0.059 | 0.639 |
| Model-M | generate | 90.57 | 237.60 | 0.381 | 0.611 |

Tiny is dominated by framework/runtime overhead and must not be used to claim general
superiority. Model-S/M show the current real gap: PyTorch is about 2.1–2.6x faster in
generation and 13–17x faster in these short training shapes.

## Official checkpoint result

| Model | Mode | microLLM token/s | PyTorch token/s | microLLM/PyTorch | Peak memory ratio |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | decode | 19.524 | 20.916 | 0.933 | 1.920 |
| Qwen2.5-0.5B | one train step | 1.571 | 0.459 | 3.423 | 0.951 |
| DeepSeek Distill 1.5B | decode | 10.252 | 31.505 | 0.325 | 1.976 |
| DeepSeek Distill 1.5B | one train step | 1.350 | 0.446 | 3.026 | 0.797 |

The inference rows compare complete `generate` calls after a separate full-logit
forward on both sides. The train rows are first-step functional measurements without a
training warm-up. Therefore the apparent 3x training advantage is not accepted as a
steady-state performance claim. A multi-step official-model benchmark is still needed.

The official one-step numerical checks remained aligned while measuring:

```text
Qwen loss absolute difference       8.58e-6
DeepSeek loss absolute difference   2.86e-6
Qwen final_norm[0] after update     7.468739033 on both paths
DeepSeek final_norm[0] after update 2.124989748 on both paths
```

The inference memory gap supports the existing hypothesis that microLLM's external
weight loading and dynamic cache path retain too much temporary data. DeepSeek decode
throughput also makes device-native preallocated K/V cache the next measured target.

## Evidence boundary

- built-in weights are independently initialized but architecture/workload shapes are
  identical; separate alignment tests cover same-weight numerical correctness;
- official rows use identical weights and token IDs and require exact generated IDs;
- allocator memory ratios are diagnostic, not total board peak;
- all rows use FP32 compute; BF16/FP8 comparison remains unproven.
