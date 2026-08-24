# Experiment 189 — faster QK/PV solutions do not move both models

Status: `keep` exact registry infrastructure; `discard` default solution policy

## Question

Experiment 188 found faster vendor solutions for all four official T512 QK/PV shapes.
Do those isolated gains remain correct and exceed 1.01 end to end when every Transformer
block uses them?

## Exact registry contract

The new key describes the descriptor seen by hipBLASLt, not a model name. It includes
flattened batch count, physical input/output rows and columns, contiguous batch strides,
transpose flags, alpha bits, operation mode, workspace limit, GPU architecture, HIP
runtime/driver versions and hipBLASLt version. Registration is explicit and thread-local.

The first matching call resolves the index with the installed library, checks exact
descriptor support and workspace, then caches the algorithm. A stale environment, negative
index, unsupported descriptor or larger workspace fails visibly. An absent or nonmatching
key continues through the ordinary default algorithm.

## Pilot correction

The fastest QK recommendations (`305434` Qwen, `305460` DeepSeek) passed isolated Max/RMS,
but repeated blocks produced complete-logit Max/RMS `0.07290/0.01436` and
`0.04437/0.00733`. PV alone stayed bit-exact. We therefore returned to the Experiment 188
inventory and selected the fastest bit-exact QK candidates: `311017` and `305423`.

## Formal result

The baseline keeps the accepted BF16 FFN Arena `rows>=512` policy. Three fresh processes
per model/policy use two warm-ups, five measurements and reversed even-run order.

| Model | Policy | Dispatches | Complete logits | Speedup | Peak |
|---|---|---:|---:|---:|---:|
| Qwen | QK | 168 | bit-exact | 1.0093× | unchanged |
| Qwen | PV | 168 | bit-exact | 1.0037× | unchanged |
| Qwen | both | 336 | bit-exact | 1.0082× | unchanged |
| DeepSeek | QK | 196 | bit-exact | 0.9991× | unchanged |
| DeepSeek | PV | 196 | bit-exact | 1.0029× | unchanged |
| DeepSeek | both | 392 | bit-exact | 1.0043× | unchanged |

![FP32 Attention complete-model gate](../assets/fp32-attention-model-gate.svg)

## Decision

Keep the exact, version-aware registry, public diagnostics, CLI controls, isolated
QK/PV runner and regression tests. Do not register any solution by default: none of
QK-only, PV-only or both reaches 1.01 on both pinned models. The experiment also shows
why isolated Kernel wins cannot be promoted directly—numeric differences can accumulate,
and even bit-exact faster Kernels can be too small a share of the end-to-end path.

This exact T512 solution track is saturated. A future retry needs a new mechanism such
as fusing surrounding scale/softmax/layout work into the same backend region; another
solution-index sweep is not a distinct hypothesis.

Raw evidence:
[`benchmarks/results/2026-08-24-fp32-attention-model-gate/`](../../../benchmarks/results/2026-08-24-fp32-attention-model-gate/).
