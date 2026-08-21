# Experiment 051 — official context-512 baseline

## Question

Do the Qwen/DeepSeek training paths remain correct at 512 tokens, and which subsystem
explains the gap to PyTorch?

## Formal result

Both models completed one warm-up and two measured steps in three fresh processes per
framework. Losses are finite, the observed parameter changes, load streaming remains
active and optimizer payload H2D/D2H stays zero.

| Model | microLLM | PyTorch | Throughput ratio | Peak ratio |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 812.45 tok/s | 8306.75 tok/s | 0.0978× | 1.239× |
| DeepSeek Distill 1.5B | 400.15 tok/s | 4811.60 tok/s | 0.0832× | 1.033× |

![Context-512 baseline and profile](../assets/context512-training-profile.svg)

This is a stable failure, not a support failure: the path is numerically valid but about
10–12× slower than the selected PyTorch BF16 reference.

## Qwen profile

The process-wide trace covers streaming load and three training steps. The clean Kernel
count identity is `24 layers × 3 steps = 72` Attention calls in each direction.

| Category | Calls | Kernel time | Share |
|---|---:|---:|---:|
| causal GQA backward | 72 | 985.61 ms | 50.64% |
| causal GQA forward | 72 | 269.85 ms | 13.86% |
| RMSNorm weight gradient | 147 | 144.39 ms | 7.42% |
| AdamW | 870 | 129.86 ms | 6.67% |
| bias gradient | 216 | 115.39 ms | 5.93% |
| everything else | 5,390 | 301.36 ms | 15.48% |

Attention is 64.50% of Kernel time. This also explains why the host-labelled optimizer
window grows with sequence length: optimizer synchronization waits for earlier Attention.

## Next falsifiable design

The current backward recomputes each softmax row and atomically accumulates K/V gradients
from every query row. The next candidate will separate row probability/score-gradient
generation from a non-atomic K/V reduction. It may use temporary T×T matrices only for
long sequences where the measured atomic path is dominant. The keep gate includes:

- full Q/K/V gradient equality against CPU/PyTorch;
- no future-token contribution;
- reduced Attention Kernel time and end-to-end T=512 improvement;
- explicit temporary-memory cost and no regression on T=128.
