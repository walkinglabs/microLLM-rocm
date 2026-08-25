# 2026-08-25 — scoped Autograd gradient producer

## Contract

- ordinary and zeroed accumulation targets retain their existing accumulation semantics;
- overwrite-only targets are leaf-only, contiguous, matching FP32 Tensors whose bytes are not
  read before a specialized full producer writes them;
- a generic first contribution abandons the target and restores ordinary first assignment;
- the default-off producer accepts only rank-2 non-transposed matmul right leaves;
- the first direct contribution consumes overwrite eligibility, so repeated/shared use safely
  accumulates later contributions;
- ordinary `set_grad` and `zero_grad` clear every target state.

## Evidence before the formal matrix

CPU tests cover complete left/right gradients, preserved nonzero fallback, generic fallback, and a
shared weight. HIP tests keep the target address, both gradients, and zero payload transfers.
The runner measures backward only on an already-built graph and reports exact gradients, address,
dispatch count, logical allocations, Event and wall timing for five shapes.

Pilot results already challenge the hypothesis: one logical allocation disappears, but FFN T32
reaches only 0.960x Event / 0.976x wall and head T32 0.983x / 0.985x. The route remains default-off
until the rotated matrix decides whether any exact shape clears 1.05x.
