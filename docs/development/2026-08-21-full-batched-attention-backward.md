# 2026-08-21 — fully batched saved Attention backward

## Delivered

For HIP sequences of at least 256 tokens with hipBLASLt, saved Attention backward now
uses batched GEMM for all four matrix derivatives: dP, dQ, dK and dV. The remaining
Attention-specific device calculation is causal softmax backward plus deterministic GQA
head reduction.

The short-sequence and library-free contracts remain unchanged.

## Evidence

- focused MHA/GQA T=256 forward and all-gradient comparison passed;
- Qwen/DeepSeek T=512 three-process medians improved 1.201x/1.309x;
- measured engine peak was byte-identical for both models;
- T128 fallback stayed within the 5% gate with byte-identical peak;
- the 306.63 ms saved-row Kernel disappeared;
- its replacement stage is about 122.21 ms (2.509x faster);
- full retained-process Kernel time improved 1.199x.

See the [experiment report](../optimization-log/experiments/057-full-batched-attention-backward.md)
and [raw evidence](../optimization-log/experiments/057-data/).

## Boundary

This is MI300X BF16 batch-1 context-512 evidence, not a universal shape or GPU claim.
Forward/backward causal softmax rows are now the largest Attention-specific costs.
