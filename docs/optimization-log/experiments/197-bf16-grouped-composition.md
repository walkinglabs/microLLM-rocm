# Experiment 197 — compose two exact grouped policies

Status: keep explicit composed policy; default unchanged

## Question

Grouped QKV and grouped gate/up pass separately, but they use two registries, two Arena families
and two shared-kernel caches in one thread. Do both really dispatch, preserve complete output and
deliver incremental speedup when enabled together?

## Four-policy matrix

Three fresh processes per model rotate baseline, QKV-only, gate/up-only and both.

| Model | Baseline | QKV | Gate/up | Both | Both/base | Both/QKV |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 93565 | 97741 | 95218 | 99690 tok/s | 1.0655× | 1.0199× |
| DeepSeek | 50328 | 51819 | 50917 | 52711 tok/s | 1.0474× | 1.0172× |

Both policies report exactly 168/196 dispatches in a combined process. A disabled side always
reports zero, which proves the composition result is not one policy silently shadowing the other.

![Grouped policy composition](../assets/bf16-grouped-composition.svg)

## Accuracy, memory and setup

All 24 complete outputs preserve top-1. Qwen Max/RMS is 0.12031/0.02905 and DeepSeek is
0.07200/0.01255, inside the established BF16 envelope. Combined peak ratios are
1.00342×/1.00173×.

Combined kernel setup is 214.5/205.6 ms. QKV accounts for almost all of it; after QKV initializes
the library path, gate/up setup falls to 0.249/0.239 ms. This is useful lifecycle interaction, but
the setup remains explicit and cannot be hidden from one-shot latency.

## Decision

Keep the explicit combination. Do not compile backend-local indices into defaults and do not infer
support for rows other than 512. Serving may combine QKV prewarm with gate/up's sub-millisecond
follow-up plan creation before admission.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-grouped-composition/](../../../benchmarks/results/2026-08-24-bf16-grouped-composition/).
