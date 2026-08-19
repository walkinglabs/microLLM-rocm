# 2026-08-19 — M3 Attention graph primitives

## Contract

Add only the graph primitives missing for readable causal Attention. Do not introduce
a fused Attention operator before the composed path is testable.

## Implementation

- `causal_softmax` accepts tensors whose final two dimensions are square;
- forward normalizes only keys at or before each query position and writes future
  probabilities as zero;
- backward applies the Softmax Jacobian-vector product only over visible positions;
- `contiguous(Value)` materializes logical order and returns gradients in the same
  logical order to a potentially non-contiguous parent.

## Verification

- all future probabilities and gradients are exactly zero in a hand-written 3x3
  causal example;
- every visible row sums to one;
- transpose → contiguous → weighted sum returns gradients to the original layout in
  hand-calculated order.

These primitives allow MHA to be expressed as matmul, scale, causal softmax, matmul,
transpose, and contiguous, preserving an inspectable backward graph.
