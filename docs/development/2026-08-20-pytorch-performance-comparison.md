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
| Qwen2.5-0.5B | generate | 18.771 | 70.182 | 0.267 | 1.225 |
| Qwen2.5-0.5B | train | 7.300 | 51.324 | 0.142 | 0.951 |
| DeepSeek Distill 1.5B | generate | 10.018 | 62.397 | 0.161 | 0.989 |
| DeepSeek Distill 1.5B | train | 5.794 | 26.226 | 0.221 | 0.797 |

Both sides first run two unmeasured warm-up iterations, reset their peak allocator
counter, and then measure five iterations. Qwen measures 20 generated or 15 trained
tokens; DeepSeek measures 40 generated or 15 trained tokens. The old first-step result
was rejected as a performance conclusion because it incorrectly suggested a 3x
microLLM training lead.

The multi-step numerical trajectories remained aligned while measuring:

```text
Qwen final loss absolute difference       5.43e-6
DeepSeek final loss absolute difference   1.14e-4
Qwen final_norm[0] after update           7.468698978 on both paths
DeepSeek final_norm[0] after update       2.125010967 on both paths
```

After resetting peak memory following warm-up, DeepSeek inference allocator peaks are
nearly equal, while microLLM Qwen remains 1.225x PyTorch allocated peak. The 3.7–6.2x
generation and 4.5–7.0x training throughput gaps make fused/device-native Attention,
preallocated K/V cache and optimized backward the next measured targets.

## Evidence boundary

- built-in weights are independently initialized but architecture/workload shapes are
  identical; separate alignment tests cover same-weight numerical correctness;
- official rows use identical weights and token IDs and require exact generated IDs;
- allocator memory ratios are diagnostic, not total board peak;
- all rows use FP32 compute; BF16/FP8 comparison remains unproven.
