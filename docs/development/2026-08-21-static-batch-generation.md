# 2026-08-21 — static cross-request batch generation

The public inference API now provides `generate_batch()`. Prompt rows may contain different tokens
but share length and one generation configuration. One batch-aware KV Cache drives full prefill,
cached steps and device row-wise argmax.

CPU greedy/stochastic tests and HIP different-row tests match independent generation. The fixed
1/2/4/8-request benchmark reaches 2,443 token/s at HIP B8, a 7.306x gain over the serial scheduler
with 90.7% scaling efficiency and exact outputs.

This is static batching, not continuous batching: delayed arrivals, unequal lengths, cancellation
and slot refill remain unsupported. See
[Experiment 073](../optimization-log/experiments/073-static-batch-generation.md).

Final gates: full CPU/HIP 275/275, ASan/UBSan 189/189 and PyTorch-enabled CPU 194/194.
