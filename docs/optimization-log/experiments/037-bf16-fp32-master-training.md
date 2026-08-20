# Experiment 037 — BF16 Linear training with FP32 masters

Status: `partial keep` — correctness/PyTorch gates pass; self-FP32 speed and memory gates fail

## Training and inference are different ownership problems

Inference can destroy FP32 weights and retain one BF16 representation. Training cannot:
AdamW must update a precise master parameter, and gradients/moments need their own state.

`LinearPrecision::BFloat16` therefore means:

```text
FP32 master activation + FP32 master weight
            ↓ round operands for forward
         BF16 GEMM, FP32 output
            ↓ straight-through backward
FP32 activation gradient + FP32 weight gradient
            ↓
         FP32 AdamW update
```

It is mutually exclusive with one-way frozen inference preparation. Model summary and the
official train CLI expose the policy explicitly as `bf16_linear_fp32_master`.

## Correctness evidence

- independent Python custom-autograd BF16 STE rebuild of the full tiny Transformer;
- logits, loss and every parameter gradient compared with microLLM;
- CPU 20-step loss decrease and FP32 parameter assertions;
- HIP full forward/backward with zero H2D/D2H payload during graph execution;
- official Qwen/DeepSeek 2 warm-up + 5 measured-step finite update trajectories;
- device-native FP32 AdamW remains unchanged.

```text
CPU CTest              167/167 pass
ASan/UBSan             165/165 pass
HIP CTest               65/65 pass
Python/PyTorch oracle     4/4 pass
official matrix          18/18 finite parameter updates
```

## Reference choice and a useful failure

Loading PyTorch parameters themselves as BF16 caused the observed Qwen parameter to remain
unchanged at learning rate `1e-5`: the update rounded away. That failed row is preserved.
The matched reference uses PyTorch BF16 autocast with FP32 parameters and AdamW masters.

## Three-process result

| Model | micro FP32 | micro BF16 master | vs micro FP32 | PyTorch BF16 AMP | micro/PyTorch | Peak BF16/FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 150.97 tok/s | 138.66 tok/s | 0.918× | 44.41 tok/s | 3.122× | 1.000× |
| DeepSeek Distill 1.5B | 81.72 tok/s | 74.06 tok/s | 0.906× | 28.68 tok/s | 2.583× | 1.000× |

![BF16 FP32-master training](../assets/bf16-training.svg)

Loss medians:

```text
Qwen micro BF16       1.00346 → 0.00249
Qwen PyTorch AMP      1.09570 → 0.00334
DeepSeek micro BF16  10.48939 → 9.26878
DeepSeek PyTorch AMP 10.46464 → 9.29171
```

Qwen/DeepSeek microLLM peak engine bytes are 9,012,293,536 and 28,468,424,608 for both
FP32 and BF16. The policy rounds compute operands but still stores FP32 parameters,
gradients and optimizer states, so no memory reduction was expected or observed.

## Decision

Keep the training policy, CLI and full-graph tests: they provide real BF16 training with
FP32 masters and exceed the matched PyTorch BF16 autocast throughput.

Do not describe BF16 as an internal optimization yet. It is `8.2%/9.4%` slower than the
retained microLLM FP32 training path and saves no peak memory. The next candidate must keep
compatible forward activations in BF16 across adjacent Linear/SwiGLU operations or define a
per-step forward-weight lifecycle; it may not weaken FP32 master/gradient/update evidence.
