# Experiment 041 — continuous BF16 FFN training island

Status: `discard` — correct gradients and fewer allocations, no stable speed win

## Hypothesis

With Experiment 040 mirrors, each FFN still casts its FP32 activation before gate, up and
down. The candidate casts once at the FFN entrance, keeps gate/up/SwiGLU activations BF16,
and returns to FP32 at the residual boundary.

Backward was not guessed. A dedicated BF16-input/FP32-gradient SwiGLU Kernel and one graph
node returned FP32 gradients to input plus gate/up/down masters. The tiny full Transformer
matched the PyTorch oracle for logits, loss and every parameter gradient. CPU, sanitizer
and HIP focused tests passed without measured host payload transfers.

## What first looked wrong

Three Qwen candidate processes produced only `18.74–18.92 token/s`, while the previously
published Experiment 040 median was `151.69 token/s`. Calling that an 8× candidate
regression would have been wrong.

The profiler showed unrelated kernels slowing together: FP32 transpose-backward, AdamW,
fill and cast all became much slower. A same-window Experiment 040 control then measured
only `18.685 token/s`. The shared MI300X execution window had changed, so old and new
absolute numbers were not comparable.

## Valid same-window decision

| Qwen measurement | Throughput | Peak bytes | Allocation calls |
|---|---:|---:|---:|
| 040 mirror control | 18.685 tok/s | 9,727,946,656 | 10,160 |
| FFN island median (3 processes) | 18.892 tok/s | 9,721,642,912 | 10,040 |
| Candidate/control | 1.011× | 0.999× | 0.988× |

![BF16 FFN training island discarded](../assets/bf16-training-ffn-island-discard.svg)

The island removes a net 120 allocations over five Qwen steps and preserves finite
updates, but improves same-window throughput by only 1.1%, below the mandatory 5% gate.
DeepSeek was stopped after its first process exceeded three minutes without completing;
once Qwen had already failed the keep gate, spending two more long processes could not
change the decision.

## Decision

Remove the public graph API, mixed backward Kernel, model switch, CLI flag and all candidate
tests. Retain only this report, the early-stop record and aggregated profiler evidence.

The failure does not prove activation islands can never work. It proves this eager island,
which still casts saved activation for the down-weight gradient and runs ordinary FP32
backward GEMMs, is not worth its graph complexity. A retry needs a larger backward contract
or a stable dedicated GPU window; merely rebuilding this exact node is prohibited.
