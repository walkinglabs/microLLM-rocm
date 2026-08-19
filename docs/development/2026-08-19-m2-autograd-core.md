# 2026-08-19 — M2 eager autograd core

## Contract

Implement eager reverse-mode autograd as a module above Tensor and Ops. Tensor stays
a data/metadata type. Autograd identity belongs to a shared `Value::Node`. The first
path supports CPU float32 and must catch gradient overwrite on graph branches.

## Implementation

- forward operations create nodes with parent references and backward closures;
- backward performs a deduplicated topological traversal;
- leaf gradients accumulate when a parameter appears on multiple paths;
- intermediate gradients reset between repeated backward calls so stale graph state
  is not propagated again;
- explicit seed gradients are shape-checked;
- `zero_grad` and `detach` are explicit;
- implemented differentiable operations: add, multiply, scale, batched matmul, sum,
  and mean.

## Verification

```text
CPU Debug complete suite:       35/35 passed before repeated-backward case
CPU ASan/UBSan complete suite:  35/35 passed
Autograd focused final suite:    6/6 passed
HIP-enabled build, CPU autograd: 5/5 passed before repeated-backward case
```

Evidence includes a hand-calculated branch graph, hand-calculated matmul gradients,
central finite differences, invalid seed shapes, and two backward calls through the
same graph.

## Failure found

The first repeated-backward design retained intermediate gradients. A second call
would therefore send the old plus new intermediate gradient to leaves and overcount.
The engine now clears non-leaf gradients before each traversal while preserving leaf
accumulation.

## Remaining M2 work

- Embedding, Softmax/cross-entropy, RMSNorm, SiLU/SwiGLU, and RoPE backward;
- SGD and AdamW;
- checkpoint format, optimizer/RNG/data cursor, and resume equivalence;
- N2 character or tiny classifier training artifact.
