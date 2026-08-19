# 2026-08-19 — M3 tiny Transformer overfit

## Contract

Connect TokenDataset, Transformer loss, eager backward, AdamW, zero-grad, and metrics
into a complete training step. Prove learnability on deterministic generated tokens
before selecting a real corpus or scaling Model-S.

## Implementation

`train_step`:

1. validates input/target shape;
2. clears parameter gradients;
3. computes language-model cross entropy;
4. rejects non-finite loss;
5. runs backward;
6. measures global gradient L2 norm;
7. applies AdamW;
8. returns step, loss, and gradient norm.

## Measured smoke trajectory

An 8-dimensional, one-layer, two-query-head/one-KV-head Transformer trained on a
repeating four-token sequence:

```text
step=1  loss=1.81171   grad_norm=24.5515
step=10 loss=0.468095  grad_norm=1.06034
step=20 loss=0.0731425 grad_norm=0.34409
step=30 loss=0.0172942 grad_norm=0.092913
step=40 loss=0.00673309 grad_norm=0.027997
```

The executable fails unless final loss is finite and below 35% of first-step loss;
the observed result is substantially below that gate. Separate trainer tests verify
finite metrics, loss reduction, and mismatched batch rejection.

## Boundary

This is an overfit smoke, not a Model-S quality result. It uses generated data,
CPU float32 backward, no validation split, and no throughput claim.
