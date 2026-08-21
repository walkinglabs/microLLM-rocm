# Experiment 045 — DeepSeek shape baseline and device-native checkpoint load

Status: `keep` for loading architecture; `baseline` for DeepSeek shapes

## Why loading became the first failure

The first DeepSeek pilot spent roughly 6–7 minutes in each microLLM process before a
sub-second measured region. Inspection found three avoidable costs:

1. construct and randomly initialize 1.78B parameters that will immediately be overwritten;
2. copy those unused values when moving the empty model to GPU;
3. load weights on CPU, transpose large Linear weights on CPU, call `to_vector()`, then copy
   them back to GPU.

The new `ParameterInitialization::Uninitialized` model rejects forward until a complete
strict load. Moving it to a device allocates storage without copying garbage. Safetensors
are loaded to the model device, Linear transpose runs on GPU, and identity copies stay
device-to-device. External StateDict ownership remains independent and strict load remains
atomic.

DeepSeek now reports a stable `load_ms` median of about 64.9–65.4 seconds across all shapes.
That is still much slower than PyTorch's 1.7–2.4 seconds, but it reduces observed process
setup from roughly 6–7 minutes to about 80 seconds and makes repeated research feasible.

## Formal DeepSeek matrix

Each row is the median of three fresh processes per framework, one warm-up and two measured
updates, BF16 Linear forward with FP32 masters.

| Shape B×T | microLLM | PyTorch | Throughput ratio | Peak-memory ratio |
|---|---:|---:|---:|---:|
| 1×3 | 14.39 tok/s | 28.26 tok/s | 0.509× | 0.882× |
| 2×3 | 28.45 tok/s | 53.49 tok/s | 0.532× | 0.883× |
| 1×32 | 140.60 tok/s | 307.63 tok/s | 0.457× | 0.891× |
| 1×128 | 395.71 tok/s | 1173.14 tok/s | 0.337× | 0.919× |

![DeepSeek training shapes and load time](../assets/deepseek-training-shapes.svg)

All 24 formal rows have finite loss, exact trained-token counts and changed parameters.
microLLM optimizer measurement regions contain zero Tensor payload transfers. Batch 2 at
context 3 scales almost exactly 2× for microLLM and about 1.89× for PyTorch.

## What this proves and does not prove

It proves the retained weight-gradient and fused Attention paths work on the 1.5B
DeepSeek-Distill-Qwen architecture, not only Qwen 0.5B. microLLM uses 8%–12% less measured
peak memory in all four rows.

It does not prove parity. The ratio declines to 0.337× at context 128. The load path is also
still around 30× slower than PyTorch. Next work separates these two problems: streaming or
mapped safetensors for setup, and retained-step profiling for training throughput.
