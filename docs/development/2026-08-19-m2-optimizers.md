# 2026-08-19 — M2 SGD and AdamW

## Contract

Add explicit parameter optimizers above `autograd::Value`. The first implementation
updates contiguous CPU float32 parameters, skips parameters without a gradient,
rejects malformed state, and makes all state required for the next AdamW step
exportable and restorable.

## Implementation

- SGD with optional coupled L2 weight decay;
- AdamW with bias correction and decoupled weight decay;
- parameter-list `zero_grad`;
- `AdamWState` containing step, first moments, and second moments;
- deep state snapshots so restoring one optimizer cannot alias another optimizer's
  moment buffers;
- shape, dtype, device, hyperparameter, and parameter-count validation.

## Verification

Focused optimizer tests pass 4/4:

- SGD lowers a scalar quadratic loss;
- AdamW's first update matches a hand-calculated bias-corrected update;
- an optimizer restored after step one produces exactly the same step-two parameters
  and step counter as uninterrupted execution;
- constants and incomplete state are rejected.

## Boundary

This proves in-memory training-state continuation. The next change defines a
versioned file format and proves process-style save/load continuation including
model parameters and caller-owned experiment state.
