# Step 35 — shared BF16 training activation casts

Status: complete, all model routes rejected, primitives retained

## Question

Can Q/K/V share one FP32-to-BF16 activation cast, and can gate/up do the same, without changing
their independent backward edges?

## Contract

- both multi-output primitives keep one graph node per output and weight;
- compare every output plus input and every weight gradient against composed BF16 matmuls;
- run a direct PyTorch oracle and a device-native HIP graph test;
- isolate QKV-only, gate/up-only and their combination;
- keep a model route only when both official B1/T512 medians reach `1.01×`.

## Decision

Reject all model routes. The five-run combined result is `1.0066×/1.0179×`; QKV-only is
`0.9804×/1.0039×`; gate/up-only is `0.9911×/1.0012×`. Retain the independently tested tensor
and Autograd primitives for a future graph compiler, not the current eager model.
