# Experiment 167 — remove scale kernels, fail the joint model gate

## Hypothesis

The retained BTHD Attention graph explicitly computes:

```text
scaled_query = query * scale
scores = scaled_query @ keyᵀ

score_gradient = softmax_backward(...) * scale
dQ = score_gradient @ key
dK = score_gradientᵀ @ query
```

hipBLASLt already accepts an alpha scalar. A scaled matmul can remove one forward and one
backward scale allocation/Kernel per layer. The readable path remains ordinary
`scale(matmul(...))`; HIP applies alpha inside GEMM.

Moving scale after accumulation changes FP32 rounding order. Therefore the gate compares
full outputs/gradients with PyTorch tolerance and requires the official observed parameter
to remain equal, not merely finite.

## Correctness boundary

- `matmul_scaled_with_implementation` covers transpose and rejects nonfinite alpha;
- CPU matches the composed scale+matmul reference;
- Python/PyTorch independently computes `(left @ right) * factor`;
- HIP alpha output matches CPU with zero payload transfer;
- T256 BTHD saved Attention output/probabilities/Q/K/V gradients match CPU;
- the false switch restores one shared backward scale, not two duplicate scales.

## Same-binary T512 result

RoPE/context fusions remain true and the rejected plan cache remains false. Only
`--attention-gemm-scale-fusion` changes. Each model/policy uses three fresh processes,
one warm-up and two measured steps.

| Model | Explicit | GEMM alpha | Speedup | Allocations saved | Peak saved | Parameter equal |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 15,264.34 | 15,064.58 tok/s | 0.9869× | 96 | 12,320,768 B | yes |
| DeepSeek Distill 1.5B | 6,337.80 | 6,405.31 tok/s | 1.0107× | 112 | 0 | no |

Loss relative differences are `0.1068%` and `0.0101%`, both inside the 0.5% numerical
gate. DeepSeek's fixed parameter changes from `2.124970913` to `2.124971151`, which is an
expected consequence of reordered rounding but fails the declared equality gate.

Qwen rocprofv3 confirms the mechanism: across three executed steps scale calls fall
`144→0`, dispatches `6,907→6,761`, and total Kernel time `110.668→110.039 ms`. This local
success does not overturn Qwen's uninstrumented throughput regression.

![Attention GEMM scale fusion discarded](../assets/attention-gemm-scale-fusion-discard.svg)

## Decision

Reject as a production Attention policy and set engine/CLI defaults false. Keep the generic
scaled-matmul primitive and explicit policy for diagnosis. A future proposal cannot simply
move the same scale between operands: that search space now has a numerical-order and
end-to-end counterexample.

Raw evidence is in
[`benchmarks/results/2026-08-23-attention-gemm-scale-fusion/`](../../../benchmarks/results/2026-08-23-attention-gemm-scale-fusion/).
