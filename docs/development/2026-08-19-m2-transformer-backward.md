# 2026-08-19 — M2 Transformer backward paths

## Contract

Extend the eager graph with the forward operations required by a Decoder-only
Transformer while retaining CPU float32 as the gradient oracle. Each nonlinear
backward requires a hand-derived formula and either finite-difference or invariant
evidence.

## Added differentiable operations

- reshape and transpose;
- Embedding with repeated-index scatter accumulation;
- Softmax Jacobian-vector product;
- RMSNorm input and weight gradients;
- SiLU and SwiGLU gradients;
- inverse-rotation RoPE gradient;
- stable mean cross-entropy gradient.

## Verification

The focused Autograd suite passes 11/11. Evidence includes:

- repeated embedding indices accumulating twice into the same row;
- cross-entropy central finite differences for every logit;
- RMSNorm central finite differences;
- SwiGLU derivative identity;
- Softmax sum invariant;
- RoPE central finite differences for every input component;
- shape preservation through reshape and transpose.

## Boundary

The backward implementation is currently the readable CPU truth path. HIP backward
kernels are not claimed. Attention will first be composed from these graph operations
and verified on an extremely small configuration before fused/tuned paths are added.
