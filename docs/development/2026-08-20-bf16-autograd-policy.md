# 2026-08-20 — BF16 autograd kept, model policy rejected

## What changed on `main`

The framework gained one narrow training primitive:

- `autograd::bf16_matmul(left, right)` requires FP32 master tensors;
- forward rounds both operands to BF16 and returns FP32 accumulation/output;
- backward computes both gradients from the FP32 master tensors;
- C++ graph tests and an independent Python/PyTorch oracle cover output, left gradient
  and right gradient.

This does not switch the Transformer to BF16 and does not change the FP32 default.

## What was tried and removed

An experimental exact-shape policy cached BF16 weights for the two M=1 shapes that had
won their micro-benchmark. Three official-model inference processes showed:

```text
Qwen2.5-0.5B               0.850× retained FP32 throughput, +73.5 MiB
DeepSeek Distill 1.5B      0.972× retained FP32 throughput, +1.44 GiB
greedy generated token IDs exact in both cases
```

The precision enum, cache and CLI switch were removed. The raw JSONL and generated
figure remain under `docs/optimization-log/experiments/015-data/` so the failed design
cannot quietly reappear later.

## Verification

```text
CPU debug                         154/154 pass
CPU ASan/UBSan                    152/152 pass
MI300X/gfx942 HIP                  55/55 pass
Python/PyTorch operator+graph       4/4 pass
```

## Next design constraint

A future BF16 model path must avoid a permanent FP32+BF16 weight duplicate and must keep
activations in BF16 across compatible chains. Recasting a FP32 activation at every
selected Linear is not an acceptable architecture.
